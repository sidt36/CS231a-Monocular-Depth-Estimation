import os
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision.ops import box_convert
import time
import gc
from typing import Tuple, List, Dict, Union, Optional
from dataclasses import dataclass

# Import for optimization
# Acknowledgement for CS231A : The quantization elements of the code have been wirtten with the help of Claude.
# The dependencies are GroundingDINO and Segment Anything Model (SAM)
try:
    import onnxruntime as ort
except ImportError:
    ort = None

try:
    import torch.quantization
    from torch.quantization import quantize_dynamic
except ImportError:
    quantize_dynamic = None

@dataclass
class MaskPrediction:
    masks: np.ndarray  # Shape: [N, H, W] boolean masks
    boxes: np.ndarray  # Shape: [N, 4] bounding boxes in xyxy format
    scores: np.ndarray  # Shape: [N] confidence scores
    labels: List[str]  # Length: N, text labels
    
    @property
    def count(self) -> int:
        return len(self.labels)

class GroundingSAM:
    """
    A class that combines Grounding DINO for text-based object detection and 
    Segment Anything Model (SAM) for instance segmentation.
    
    Supports both CPU and GPU inference with various optimizations.
    """
    
    def __init__(
        self, 
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_checkpoint_path: str,
        device: str = None,
        optimization_level: str = "medium",  # Options: "none", "light", "medium", "full"
        sam_model_type: str = "vit_h",  # Options: "vit_h" (default), "vit_l", "vit_b" (fastest)
        box_threshold: float = 0.25,
        text_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        multimask_output: bool = True
    ):
        """
        Initialize the GroundingSAM model.
        
        Args:
            grounding_dino_config_path: Path to the Grounding DINO config file
            grounding_dino_checkpoint_path: Path to the Grounding DINO checkpoint file
            sam_checkpoint_path: Path to the SAM checkpoint file
            device: Device to run inference on ("cuda", "cuda:0", "cpu", etc.). 
                    If None, will use cuda if available, else cpu.
            optimization_level: Level of optimization to apply:
                                "none": No optimization
                                "light": Basic optimizations (FP16, etc.)
                                "medium": More optimizations (quantization)
                                "full": Maximum optimization (ONNX, etc.)
            sam_model_type: SAM model type to use ("vit_h", "vit_l", "vit_b")
            box_threshold: Confidence threshold for bounding box detection
            text_threshold: Confidence threshold for text matching
            iou_threshold: IoU threshold for box filtering
            multimask_output: Whether to output multiple masks per detection
        """
        # Store parameters
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold
        self.iou_threshold = iou_threshold
        self.multimask_output = multimask_output
        self.optimization_level = optimization_level
        
        # Determine device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Track if we're using cuda for cleanup later
        self.using_cuda = self.device.startswith("cuda")
            
        # Print setup information
        print(f"Initializing GroundingSAM with device: {self.device}")
        print(f"Optimization level: {optimization_level}")
        
        # Initialize models
        self._initialize_models(
            grounding_dino_config_path,
            grounding_dino_checkpoint_path,
            sam_checkpoint_path,
            sam_model_type
        )
        
        # Apply optimizations based on the selected level
        self._apply_optimizations()
        
        # Keep track of timing
        self.timings = {
            "grounding_dino": [],
            "sam": [],
            "total": []
        }
        
        print("GroundingSAM initialized and ready for inference.")

    def _initialize_models(
        self,
        grounding_dino_config_path: str,
        grounding_dino_checkpoint_path: str,
        sam_checkpoint_path: str,
        sam_model_type: str
    ):
        """Initialize both models with proper error handling."""
        
        # Initialize Grounding DINO
        print("Loading Grounding DINO...")
        try:
            from groundingdino.util.inference import load_model
            self.grounding_dino_model = load_model(grounding_dino_config_path, grounding_dino_checkpoint_path)
            self.grounding_dino_model.to(self.device)
            print("Grounding DINO loaded successfully.")
            
            # Initialize our custom grounding DINO preprocessing for loading images
            from groundingdino.util.inference import load_image
            self.load_image = load_image
        except Exception as e:
            print(f"Error loading Grounding DINO: {e}")
            raise
            
        # Initialize SAM
        print(f"Loading SAM ({sam_model_type})...")
        try:
            from segment_anything import sam_model_registry, SamPredictor
            self.sam_model = sam_model_registry[sam_model_type](checkpoint=sam_checkpoint_path)
            self.sam_model.to(self.device)
            self.sam_predictor = SamPredictor(self.sam_model)
            print("SAM loaded successfully.")
        except Exception as e:
            print(f"Error loading SAM: {e}")
            raise
            
    def _apply_optimizations(self):
        """Apply optimizations based on the selected level."""
        
        # Base case - no optimizations
        if self.optimization_level == "none":
            print("No optimizations applied.")
            return
            
        # Light optimizations - applicable to both CPU and GPU
        if self.optimization_level in ["light", "medium", "full"]:
            # Use mixed precision for CUDA if available
            if self.using_cuda and hasattr(torch.cuda, 'amp'):
                print("Enabling mixed precision (FP16)...")
                self.mixed_precision = True
            else:
                self.mixed_precision = False
                
            # Set inference mode
            print("Enabling torch inference mode...")
            self._inference_mode = True
                
        # Medium optimizations - including quantization for CPU
        if self.optimization_level in ["medium", "full"]:
            if not self.using_cuda and quantize_dynamic is not None:
                print("Applying dynamic quantization to models...")
                # Quantize models for CPU
                try:
                    # Quantize Grounding DINO
                    self.grounding_dino_model = quantize_dynamic(
                        self.grounding_dino_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    print("Grounding DINO quantized successfully.")
                except Exception as e:
                    print(f"Error quantizing Grounding DINO: {e}")
                
                try:
                    # Quantize SAM
                    self.sam_model = quantize_dynamic(
                        self.sam_model, {torch.nn.Linear}, dtype=torch.qint8
                    )
                    # Need to update the predictor with the quantized model
                    self.sam_predictor.model = self.sam_model
                    print("SAM quantized successfully.")
                except Exception as e:
                    print(f"Error quantizing SAM: {e}")
            
            # For CUDA, optimize with torch.compile if available (PyTorch 2.0+)
            elif self.using_cuda and hasattr(torch, 'compile'):
                try:
                    print("Applying torch.compile to models...")
                    self.grounding_dino_model = torch.compile(self.grounding_dino_model)
                    self.sam_model = torch.compile(self.sam_model)
                    print("Models compiled successfully.")
                except Exception as e:
                    print(f"Error compiling models: {e}")
                
        # Full optimizations - ONNX runtime when available
        if self.optimization_level == "full" and ort is not None:
            print("Full optimization with ONNX runtime not implemented in this version.")
            # Note: Full ONNX optimization would require model export and is beyond
            # the scope of this implementation
            
    def _preprocess_caption(self, caption: str) -> str:
        """Preprocess caption for Grounding DINO."""
        result = caption.lower().strip()
        if result.endswith("."):
            return result
        return result + "."
    
    def _custom_predict(
        self,
        image: torch.Tensor,
        caption: str,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """Custom predict function that works with both CPU and GPU."""
        
        # Preprocess caption
        caption = self._preprocess_caption(caption)
        
        # Make sure the model and image are on the correct device
        image = image.to(self.device)
        
        # Run inference with appropriate optimizations
        if self.mixed_precision and self.using_cuda:
            with torch.cuda.amp.autocast():
                with torch.no_grad():
                    outputs = self.grounding_dino_model(image[None], captions=[caption])
        else:
            with torch.no_grad():
                outputs = self.grounding_dino_model(image[None], captions=[caption])
        
        # Extract predictions
        logits = outputs["pred_logits"].cpu().sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"].cpu()[0]  # (nq, 4)
        
        # Filter predictions
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        phrases = []
        
        for logit, box in zip(logits_filt, boxes_filt):
            phrases.append(caption[:-1])  # Remove the period at the end
        
        # Filter out low-scoring predictions
        filt_mask = logits_filt.max(dim=1)[0] > self.box_threshold
        logits_filt = logits_filt[filt_mask]
        boxes_filt = boxes_filt[filt_mask]
        phrases = [phrases[i] for i, keep in enumerate(filt_mask) if keep]
        
        return boxes_filt, logits_filt, phrases
    
    def _nms(self, boxes, scores, iou_threshold=0.5):
        """Non-maximum suppression to filter out overlapping boxes."""
        # Sort by score
        sorted_indices = torch.argsort(scores, descending=True)
        
        keep_indices = []
        while sorted_indices.size(0) > 0:
            # Pick the box with highest score
            i = sorted_indices[0].item()
            keep_indices.append(i)
            
            # Compute IoU of the picked box with respect to the rest
            if sorted_indices.size(0) == 1:
                break
                
            remaining_indices = sorted_indices[1:]
            remaining_boxes = boxes[remaining_indices]
            
            # Convert to xmin, ymin, xmax, ymax
            box_i = boxes[i].unsqueeze(0)
            
            # Compute IoU
            # Area of intersection
            tl = torch.max(box_i[:, :2], remaining_boxes[:, :2])
            br = torch.min(box_i[:, 2:], remaining_boxes[:, 2:])
            wh = (br - tl).clamp(min=0)
            intersection = wh[:, 0] * wh[:, 1]
            
            # Area of both boxes
            area_i = (box_i[:, 2] - box_i[:, 0]) * (box_i[:, 3] - box_i[:, 1])
            area_remaining = (remaining_boxes[:, 2] - remaining_boxes[:, 0]) * (remaining_boxes[:, 3] - remaining_boxes[:, 1])
            
            # IoU = intersection / union
            union = area_i + area_remaining - intersection
            iou = intersection / union
            
            # Filter out boxes with IoU > threshold
            mask = iou <= iou_threshold
            sorted_indices = remaining_indices[mask]
        
        return keep_indices
    
    def predict_mask(
        self, 
        text_prompt: str, 
        image_path: Optional[str] = None,
        image: Optional[np.ndarray] = None,
        visualize: bool = False,
        return_visualization: bool = False
    ) -> Union[MaskPrediction, Tuple[MaskPrediction, np.ndarray]]:
        """
        Predict masks based on text prompt and image.
        
        Args:
            text_prompt: Text description of objects to find
            image_path: Path to the image file (if image is not provided)
            image: BGR image as numpy array (if image_path is not provided)
            visualize: Whether to visualize the results
            return_visualization: Whether to return the visualization as a numpy array
            
        Returns:
            MaskPrediction object containing masks, boxes, scores, and labels
            If return_visualization=True, also returns visualization as numpy array
        """
        # Check that we have either image_path or image
        if image_path is None and image is None:
            raise ValueError("Either image_path or image must be provided")
            
        # Start timing
        start_time = time.time()
        
        # Load or use provided image
        if image is not None:
            # Convert BGR to RGB if needed
            if len(image.shape) == 3 and image.shape[2] == 3:
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                image_rgb = image
                
            # Load image for Grounding DINO
            # We need to create a temp file for load_image to work
            import tempfile
            with tempfile.NamedTemporaryFile(suffix=".jpg") as f:
                cv2.imwrite(f.name, image)
                image_source, tensor_image = self.load_image(f.name)
        else:
            # Load image from file
            image_source, tensor_image = self.load_image(image_path)
            # Also load with OpenCV for SAM
            image = cv2.imread(image_path)
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
        # Step 1: Run Grounding DINO to get bounding boxes
        grounding_start = time.time()
        try:
            boxes, logits, phrases = self._custom_predict(
                image=tensor_image,
                caption=text_prompt
            )
            
            # Run NMS to filter out overlapping boxes
            if len(boxes) > 0:
                # Get max score for each phrase
                scores = logits.max(dim=1)[0]
                # Convert boxes from cxcywh to xyxy format for NMS
                boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
                # Run NMS
                keep_indices = self._nms(boxes_xyxy, scores, iou_threshold=self.iou_threshold)
                # Filter boxes, scores, and phrases
                boxes = boxes[keep_indices]
                scores = scores[keep_indices]
                phrases = [phrases[i] for i in keep_indices]
            else:
                scores = torch.tensor([])
        except Exception as e:
            print(f"Error running Grounding DINO: {e}")
            # Return empty results
            return MaskPrediction(
                masks=np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool),
                boxes=np.zeros((0, 4)),
                scores=np.zeros(0),
                labels=[]
            )
        
        grounding_end = time.time()
        self.timings["grounding_dino"].append(grounding_end - grounding_start)
        
        # If no boxes found, return empty result
        if len(boxes) == 0:
            print("No objects found matching the text prompt.")
            return MaskPrediction(
                masks=np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool),
                boxes=np.zeros((0, 4)),
                scores=np.zeros(0),
                labels=[]
            )
            
        # Step 2: Convert Grounding DINO boxes to SAM input format
        H, W, _ = image_rgb.shape
        boxes_xyxy = box_convert(boxes=boxes, in_fmt="cxcywh", out_fmt="xyxy")
        boxes_xyxy = boxes_xyxy * torch.Tensor([W, H, W, H])
        boxes_xyxy = boxes_xyxy.cpu().numpy()
        
        # Step 3: Run SAM to get masks
        sam_start = time.time()
        try:
            # Set the image for SAM
            self.sam_predictor.set_image(image_rgb)
            
            # Process each detected box
            all_masks = []
            all_scores = []
            
            for i, box in enumerate(boxes_xyxy):
                # Convert box to the format SAM expects
                sam_box = box.astype(int)
                
                # Generate mask for the box
                if self.mixed_precision and self.using_cuda:
                    with torch.cuda.amp.autocast():
                        with torch.no_grad():
                            masks_tensor, mask_scores, _ = self.sam_predictor.predict(
                                box=sam_box,
                                multimask_output=self.multimask_output
                            )
                else:
                    with torch.no_grad():
                        masks_tensor, mask_scores, _ = self.sam_predictor.predict(
                            box=sam_box,
                            multimask_output=self.multimask_output
                        )
                
                # Get the highest scoring mask
                best_mask_idx = np.argmax(mask_scores)
                all_masks.append(masks_tensor[best_mask_idx])
                all_scores.append(mask_scores[best_mask_idx].item())
        
        except Exception as e:
            print(f"Error running SAM: {e}")
            # Return boxes without masks
            return MaskPrediction(
                masks=np.zeros((len(boxes_xyxy), image_rgb.shape[0], image_rgb.shape[1]), dtype=bool),
                boxes=boxes_xyxy,
                scores=scores.numpy(),
                labels=phrases
            )
            
        sam_end = time.time()
        self.timings["sam"].append(sam_end - sam_start)
        
        # Step 4: Prepare results
        if len(all_masks) > 0:
            masks_array = np.stack(all_masks, axis=0)
        else:
            masks_array = np.zeros((0, image_rgb.shape[0], image_rgb.shape[1]), dtype=bool)
            
        # Calculate total time
        end_time = time.time()
        total_time = end_time - start_time
        self.timings["total"].append(total_time)
        
        # Print timing info
        print(f"Detection completed in {total_time:.3f}s "
              f"(Grounding DINO: {grounding_end - grounding_start:.3f}s, "
              f"SAM: {sam_end - sam_start:.3f}s)")
        print(f"Found {len(all_masks)} objects matching '{text_prompt}'")
        
        # Create result object
        result = MaskPrediction(
            masks=masks_array,
            boxes=boxes_xyxy,
            scores=np.array(all_scores),
            labels=phrases
        )
        
        # Visualize results if requested
        if visualize or return_visualization:
            viz_image = self._visualize_predictions(image_rgb, result)
            
            if visualize:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 10))
                plt.imshow(viz_image)
                plt.axis('off')
                plt.title(f"Results for '{text_prompt}'")
                plt.show()
                
            if return_visualization:
                return result, viz_image
                
        return result
    
    def _visualize_predictions(self, image: np.ndarray, predictions: MaskPrediction) -> np.ndarray:
        """Create visualization of predictions."""
        import matplotlib.pyplot as plt
        from matplotlib.figure import Figure
        from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
        
        # Create figure and axes
        fig = Figure(figsize=(10, 10))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)
        
        # Show image
        ax.imshow(image)
        ax.set_title(f"Detection Results ({len(predictions.labels)} objects)")
        ax.axis('off')
        
        # Show masks with random colors
        for i, (mask, box, score, label) in enumerate(zip(
            predictions.masks, predictions.boxes, predictions.scores, predictions.labels
        )):
            # Generate random color
            color = np.concatenate([np.random.random(3), [0.5]])
            
            # Show mask
            h, w = mask.shape
            mask_img = np.ones((h, w, 4))
            mask_img[:, :, :3] = color[:3]
            mask_img[:, :, 3] = mask * 0.5  # Use mask as alpha channel
            ax.imshow(mask_img)
            
            # Show bounding box
            rect = plt.Rectangle(
                (box[0], box[1]),
                box[2] - box[0],
                box[3] - box[1],
                linewidth=2,
                edgecolor=color[:3],
                facecolor='none'
            )
            ax.add_patch(rect)
            
            # Add text label
            ax.text(
                box[0], box[1] - 5,
                f"{label} ({score:.2f})",
                color='white', fontsize=10,
                bbox=dict(facecolor=color[:3], alpha=0.5)
            )
            
        # Convert figure to image
        fig.tight_layout()
        canvas.draw()
        viz_image = np.array(canvas.renderer.buffer_rgba())
        
        # Close figure to free memory
        plt.close(fig)
        
        return viz_image
    
    def get_average_timings(self):
        """Return average timings for each component."""
        return {
            "grounding_dino": np.mean(self.timings["grounding_dino"]) if self.timings["grounding_dino"] else 0,
            "sam": np.mean(self.timings["sam"]) if self.timings["sam"] else 0,
            "total": np.mean(self.timings["total"]) if self.timings["total"] else 0
        }
    
    def clear_timings(self):
        """Clear timing history."""
        self.timings = {
            "grounding_dino": [],
            "sam": [],
            "total": []
        }
        
    def __del__(self):
        """Clean up resources when the object is deleted."""
        # Clear references to large models
        if hasattr(self, 'grounding_dino_model'):
            del self.grounding_dino_model
        if hasattr(self, 'sam_model'):
            del self.sam_model
        if hasattr(self, 'sam_predictor'):
            del self.sam_predictor
            
        # Perform garbage collection
        gc.collect()
        
        # Clear CUDA cache if we were using it
        if self.using_cuda:
            torch.cuda.empty_cache()


# Example usage
if __name__ == "__main__":
    # Initialize model
    ## See testbed.ipynb for tutorial
    grounding_sam = GroundingSAM(
        grounding_dino_config_path="GroundingDINO/config/GroundingDINO_SwinT_OGC.py",
        grounding_dino_checkpoint_path="models/groundingdino_swint_ogc.pth",
        sam_checkpoint_path="sam_vit_h_4b8939.pth",
        device=None,  # Auto-detect GPU if available
        optimization_level="medium",  # Options: "none", "light", "medium", "full"
        sam_model_type="vit_h"  # For faster inference on CPU, consider "vit_b"
    )
    
    # Predict masks
    results = grounding_sam.predict_mask(
        text_prompt="green block",
        image_path="sample_images/base.png",
        visualize=True
    )
    
    print(f"Found {results.count} objects")
    print(f"Average inference time: {grounding_sam.get_average_timings()['total']:.3f}s")