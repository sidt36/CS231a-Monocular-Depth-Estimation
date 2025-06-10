import torch
import numpy as np
from PIL import Image
import requests
from io import BytesIO
import matplotlib.pyplot as plt
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import time

class DepthEstimator:
    """
    A class for depth estimation using Depth Anything model.
    Loads the model once and can process multiple images efficiently.
    """
    
    def __init__(self, model_name="LiheYoung/depth-anything-base-hf", device=None):
        """
        Initialize the depth estimator with the model.
        
        Args:
            model_name: HuggingFace model name
            device: Device to run on ('cuda', 'cpu', or None for auto-detect)
        """
        print(f"Initializing DepthEstimator with {model_name}...")
        start_time = time.time()
        
        # Set device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        # Load model and processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if self.device == 'cuda' else torch.float32
        )
        self.model.to(self.device)
        self.model.eval()  # Set to evaluation mode
        
        # Store default parameters
        self.default_crop_percent = 0.05
        self.default_percentile_floor = 5
        
        print(f"Model loaded in {time.time() - start_time:.2f} seconds")
        print(f"Running on: {self.device}")
    
    def load_image(self, image_source, is_url=False):
        """
        Load an image from file path, URL, or PIL Image.
        
        Args:
            image_source: File path, URL, or PIL Image
            is_url: Whether the source is a URL
            
        Returns:
            PIL Image
        """
        if isinstance(image_source, Image.Image):
            return image_source
        elif is_url:
            return Image.open(BytesIO(requests.get(image_source, stream=True).raw.read()))
        else:
            return Image.open(image_source)
    
    def predict_depth(self, image_source, is_url=False):
        """
        Predict depth map from an image.
        
        Args:
            image_source: Image path, URL, or PIL Image
            is_url: Whether the source is a URL
            
        Returns:
            raw_depth: Raw depth array (inverse depth)
            image: Original PIL Image
        """
        # Load image
        image = self.load_image(image_source, is_url)
        
        # Process image
        inputs = self.image_processor(images=image, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get depth prediction
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Post-process
        post_processed = self.image_processor.post_process_depth_estimation(
            outputs,
            target_sizes=[(image.height, image.width)],
        )
        
        raw_depth = post_processed[0]["predicted_depth"].cpu().numpy()
        
        return raw_depth, image
    
    @staticmethod
    def get_center_crop_indices(height, width, crop_percent=0.10):
        """Get indices for center crop of an image."""
        crop_h = int(height * np.sqrt(crop_percent))
        crop_w = int(width * np.sqrt(crop_percent))
        
        y_start = (height - crop_h) // 2
        y_end = y_start + crop_h
        x_start = (width - crop_w) // 2
        x_end = x_start + crop_w
        
        return y_start, y_end, x_start, x_end
    
    def convert_to_metric(self, raw_depth, camera_height, crop_percent=None, percentile_floor=None):
        """
        Convert raw depth to metric depth using center crop calibration.
        
        Args:
            raw_depth: Raw depth array from model
            camera_height: Camera height above floor in meters
            crop_percent: Percentage of image to use for calibration
            percentile_floor: Percentile to identify floor pixels
            
        Returns:
            metric_depth: Depth in meters
            height_map: Height above floor in meters
            crop_info: Calibration information
        """
        if crop_percent is None:
            crop_percent = self.default_crop_percent
        if percentile_floor is None:
            percentile_floor = self.default_percentile_floor
            
        h, w = raw_depth.shape
        
        # Get center crop
        y_start, y_end, x_start, x_end = self.get_center_crop_indices(h, w, crop_percent)
        center_crop = raw_depth[y_start:y_end, x_start:x_end]
        
        # Find floor pixels (minimum values for inverse depth)
        floor_threshold = np.percentile(center_crop, percentile_floor)
        floor_mask_crop = center_crop <= floor_threshold
        
        # Calculate calibration
        floor_values = center_crop[floor_mask_crop]
        if len(floor_values) == 0:
            raise ValueError("No floor pixels found in center crop")
        
        avg_floor_inverse = np.mean(floor_values)
        k = camera_height * avg_floor_inverse
        
        # Convert to metric
        epsilon = 1e-8
        metric_depth = k / (raw_depth + epsilon)
        metric_depth = np.clip(metric_depth, 0, camera_height * 2)
        
        height_map = camera_height - metric_depth
        
        # Store calibration info
        crop_info = {
            'crop_bounds': (y_start, y_end, x_start, x_end),
            'crop_percent': crop_percent,
            'floor_inverse_mean': avg_floor_inverse,
            'k_constant': k,
            'camera_height': camera_height,
            'center_crop': center_crop
        }
        
        return metric_depth, height_map, crop_info
    
    def process_image(self, image_source, camera_height, is_url=False, 
                     crop_percent=None, percentile_floor=None, verbose=True):
        """
        Complete pipeline: predict depth and convert to metric.
        
        Args:
            image_source: Image path, URL, or PIL Image
            camera_height: Camera height above floor in meters
            is_url: Whether the source is a URL
            crop_percent: Percentage for center crop calibration
            percentile_floor: Percentile for floor detection
            verbose: Whether to print statistics
            
        Returns:
            Dictionary with all results
        """
        # Predict depth
        raw_depth, image = self.predict_depth(image_source, is_url)
        
        # Convert to metric
        metric_depth, height_map, crop_info = self.convert_to_metric(
            raw_depth, camera_height, crop_percent, percentile_floor
        )
        
        if verbose:
            print(f"\nProcessing complete:")
            print(f"Image size: {image.size}")
            print(f"Camera height: {camera_height:.3f}m")
            print(f"Metric depth range: {metric_depth.min():.3f}m to {metric_depth.max():.3f}m")
            print(f"Height above floor: {height_map.min():.3f}m to {height_map.max():.3f}m")
        
        return {
            'image': image,
            'raw_depth': raw_depth,
            'metric_depth': metric_depth,
            'height_map': height_map,
            'crop_info': crop_info,
            'camera_height': camera_height
        }
    
    def visualize_results(self, results, show_distribution=False):
        """
        Visualize depth estimation results.
        
        Args:
            results: Dictionary from process_image()
            show_distribution: Whether to show height distribution
        """
        image = results['image']
        metric_depth = results['metric_depth']
        height_map = results['height_map']
        crop_info = results['crop_info']
        
        # Create visualization
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Original image with crop region
        y1, y2, x1, x2 = crop_info['crop_bounds']
        axes[0, 0].imshow(image)
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, 
                           fill=False, edgecolor='red', linewidth=2)
        axes[0, 0].add_patch(rect)
        axes[0, 0].set_title("Original Image (red = calibration region)")
        axes[0, 0].axis("off")
        
        # Raw depth
        im1 = axes[0, 1].imshow(results['raw_depth'], cmap="viridis")
        axes[0, 1].set_title("Raw Depth (Inverse)")
        axes[0, 1].axis("off")
        plt.colorbar(im1, ax=axes[0, 1], fraction=0.046)
        
        # Center crop
        im2 = axes[0, 2].imshow(crop_info['center_crop'], cmap="viridis")
        axes[0, 2].set_title("Center Crop")
        axes[0, 2].axis("off")
        plt.colorbar(im2, ax=axes[0, 2], fraction=0.046)
        
        # Metric depth
        im3 = axes[1, 0].imshow(metric_depth, cmap="viridis")
        axes[1, 0].set_title("Metric Depth (m)")
        axes[1, 0].axis("off")
        plt.colorbar(im3, ax=axes[1, 0], fraction=0.046)
        
        # Height map
        im4 = axes[1, 1].imshow(height_map, cmap="plasma")
        axes[1, 1].set_title("Height Above Floor (m)")
        axes[1, 1].axis("off")
        plt.colorbar(im4, ax=axes[1, 1], fraction=0.046)
        
        # Object mask
        object_mask = height_map > 0.02
        axes[1, 2].imshow(object_mask, cmap="gray")
        axes[1, 2].set_title("Objects (>2cm)")
        axes[1, 2].axis("off")
        
        plt.tight_layout()
        plt.show()
        
        # Optional height distribution
        if show_distribution:
            self.plot_height_distribution(height_map, crop_info)
    
    def plot_height_distribution(self, height_map, crop_info):
        """Plot height distribution in workspace."""
        y1, y2, x1, x2 = crop_info['crop_bounds']
        workspace_heights = height_map[y1:y2, x1:x2].flatten()
        valid_heights = workspace_heights[(workspace_heights > 0.01) & (workspace_heights < 0.5)]
        
        if len(valid_heights) > 0:
            plt.figure(figsize=(10, 6))
            plt.hist(valid_heights, bins=50, edgecolor='black', alpha=0.7)
            plt.axvline(np.mean(valid_heights), color='red', linestyle='--', 
                       label=f'Mean: {np.mean(valid_heights):.3f}m')
            plt.axvline(np.median(valid_heights), color='green', linestyle='--', 
                       label=f'Median: {np.median(valid_heights):.3f}m')
            plt.xlabel('Height above floor (m)')
            plt.ylabel('Pixel count')
            plt.title('Height Distribution in Workspace')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
    
    def batch_process(self, image_sources, camera_heights, **kwargs):
        """
        Process multiple images efficiently.
        
        Args:
            image_sources: List of image paths/URLs/PIL Images
            camera_heights: List of camera heights or single value
            **kwargs: Additional arguments for process_image
            
        Returns:
            List of result dictionaries
        """
        if not isinstance(camera_heights, list):
            camera_heights = [camera_heights] * len(image_sources)
        
        results = []
        for i, (img_src, cam_height) in enumerate(zip(image_sources, camera_heights)):
            print(f"\nProcessing image {i+1}/{len(image_sources)}")
            result = self.process_image(img_src, cam_height, **kwargs)
            results.append(result)
        
        return results

if __name__ == "__main__":
    # Initialize once
    estimator = DepthEstimator()
    
    # Process single image
    image_path = r"C:\CS231A-Project\output_images\ee_rgb_1.png"
    results = estimator.process_image(
        image_path, 
        camera_height=0.6,
        crop_percent=0.15,
        percentile_floor=30
    )
    
    # Visualize
    estimator.visualize_results(results, show_distribution=True)
