import numpy as np
import argparse
import os
import pathlib
import torch
from itertools import chain
from matplotlib import pyplot as plt
from skimage.io import imsave
from collections import deque, defaultdict 
import json
from utils.common import load_chkpt, get_splits, draw_grasp
from env import UR5PickEnviornment
import affordance_model
from vision.segmentation_utils import get_green_mask
from skimage.measure import label, regionprops
from vision. grounded_sam import GroundingSAM
from vision.grasp_detection import predict_grasp_from_numpy, predict_grasp_angle_from_numpy
import random
############################################## 
# This script perform inference for affordance based pick and place 
# It assume that the affordance model is trained and saved

def main():
    # parse input 
    parser = argparse.ArgumentParser(description='Model eval script')
    parser.add_argument('-t', '--task', default='train',
        help='which task to do: "train", "test", "empty_bin"')
    parser.add_argument('--headless', action='store_true',
        help='launch pybullet GUI or not')
    parser.add_argument('--seed', type=int, default=3,
        help='random seed for empty_bin task')
    parser.add_argument('--save_data', default=False, action='store_true',
        help='save data for additional self-improvement on the affordance model')
    parser.add_argument('--n_past_actions', default=0, type=int)
    args = parser.parse_args()

    # load model
    model_class = affordance_model.AffordanceModel
    model_dir = os.path.join('data/affordance')
    chkpt_path = os.path.join(model_dir, 'best.ckpt')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_class(n_past_actions=args.n_past_actions)
    model.to(device)
    def load_chkpt_custom(model, chkpt_path, device):
        checkpoint = torch.load(chkpt_path, map_location=device, weights_only=False)
        # Extract the model_state_dict from the checkpoint
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
        return model



    load_chkpt_custom(model, chkpt_path, device)
    model.eval()

    detector = GroundingSAM(
                grounding_dino_config_path=r"C:\CS231A-Project\vision\GroundingDINO\config\GroundingDINO_SwinT_OGC.py",
                grounding_dino_checkpoint_path=r"C:\CS231A-Project\vision\models\groundingdino_swint_ogc.pth",
                sam_checkpoint_path=r"C:\CS231A-Project\vision\models\sam_vit_b_01ec64.pth",
                device="cpu",  # or "cuda"
                optimization_level="medium",  # Options: "none", "light", "medium", "full"
                sam_model_type="vit_b",  # For faster CPU inference, use "vit_b",
                box_threshold=0.15,  # Threshold for box detection
                text_threshold=0.15  # Threshold for text detection    

            )


        # load env
    env = UR5PickEnviornment(gui=not args.headless)
    
    if args.task == 'train' or args.task == 'test':
        names = get_splits()[args.task]
        n_attempts = 3
        vis_dir = os.path.join(model_dir, 'eval_pick_'+ args.task +'ing_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        results = list()
        for name_idx, name in enumerate(names):
            print('Picking: {}'.format(name))
            env.remove_objects()
            for i in range(n_attempts):
                print('Attempt: {}'.format(i))
                seed = name_idx * 100 + i + 10000
                if i == 0:
                    env.load_ycb_objects([name], seed=seed)
                else:
                    env.reset_objects(seed)
                
                rgb_obs, depth_obs, _ = env.observe(i)
                coord, angle, vis_img = model.predict_grasp(rgb_obs)
                pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
                result = env.execute_grasp(*pick_pose)
                print('Success!' if result else 'Failed:(')
                fname = os.path.join(vis_dir, '{}_{}.png'.format(name, i))
                imsave(fname, vis_img)
                results.append(result)
        success_rate = np.array(results, dtype=np.float32).mean()
        print("Testing on {} objects. Success rate: {}".format(args.task,success_rate))
    else:
        # names = ["YcbMustardBottle"]
        # n_attempts = 25
        # vis_dir = os.path.join(model_dir, 'eval_banana_trials')
        # pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)

        # print("Running banana-only trials.")
        # env.remove_objects()
        # count = 0
        # for attempt_id in range(n_attempts):
        #     print(f"Trial {attempt_id+1}/{n_attempts}: Placing banana at random pose.")
        #     env.remove_objects()
        #     env.load_ycb_objects(names, seed=args.seed + attempt_id)  # Different seed for each placement

        #     rgb_obs, depth_obs, _ = env.observe(attempt_id)
        #     camera_height, _ = env.get_camera_target_distance()
        #     camera_height = camera_height + 0.06  # Adjust camera height if necessary
        #     print(f"Camera height: {camera_height}")

        #     coord, angle, vis_img = model.predict_grasp(rgb_obs)
        #     temp_save_path = os.path.join(vis_dir, f'rgb_obs_{attempt_id}.png')
        #     imsave(temp_save_path, rgb_obs)
        #     print(f"Saved RGB observation to {temp_save_path}")
        #     # depth_obs = depth_model.process_image(temp_save_path, camera_height)['metric_depth']

        #     # Use detector to get banana mask
        #     mask = detector.predict_mask("mustard bottle", temp_save_path)
        #     if not mask or not hasattr(mask, 'masks') or len(mask.masks) == 0:
        #         print("No mask found.")
        #         continue
        #     else:
        #         mask = mask.masks[0]
        #         binary_mask = mask > 0
        #         labeled_mask = label(binary_mask)
        #         regions = regionprops(labeled_mask)
        #         centroids = [region.centroid for region in regions]

        #     if centroids:
        #         largest_region_idx = np.argmax([region.area for region in regions])
        #         coord = (int(centroids[largest_region_idx][1]), int(centroids[largest_region_idx][0]))
        #         print(coord)
        #         print(f"Found {len(centroids)} clusters. Using largest at {coord}")
        #         angle = predict_grasp_angle_from_numpy(mask)
        #         print(f"Predicted angle: {angle}")

        #     else:
        #         print("No banana detected")
        #         coord = (rgb_obs.shape[0] // 2, rgb_obs.shape[1] // 2)  # Default to center


        #     pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs)
        #     result = env.execute_grasp(*pick_pose)
        #     if result:
        #         env.execute_place()

        #     num_in = env.num_object_in_tote1()
        #     print(f"{1 - (num_in)}/1 banana moved")

        #     count += (1 - (num_in))

        #     fname = os.path.join(vis_dir, f'{attempt_id}.png')
        #     imsave(fname, vis_img)
        # print(f"Total moved: {count}/{n_attempts}")
        # print(f"Trials complete. {num_in} bananas left in the bin.")

        names = ["YcbBanana", 
         "YcbChipsCan",
         "YcbMustardBottle", "YcbPear", "YcbPowerDrill"]

        n_sets = 5
        attempts_per_set = 5
        vis_dir = os.path.join(model_dir, 'eval_empty_bin_vis')
        pathlib.Path(vis_dir).mkdir(parents=True, exist_ok=True)


        objects_picked_per_set = []

        for set_num in range(n_sets):
            print(f"\n{'='*50}")
            print(f"SET {set_num + 1} / {n_sets}")
            print(f"{'='*50}\n")
            
            # Reset environment for each set
            env.remove_objects()
            env.load_ycb_objects(names, seed=args.seed + set_num)
            n_objects = len(names)
            initial_objects_in_bin = env.num_object_in_tote1()
            
            object_names = ["banana", "chips can", "power drill", "mustard bottle", "pear"]
            remaining_objects = object_names.copy()
            
            for attempt_id in range(attempts_per_set):
                print(f"\nAttempt {attempt_id + 1} / {attempts_per_set}")
                
                if not remaining_objects:
                    print("All objects have been picked. Resetting list.")
                    remaining_objects = object_names.copy()
                
                rgb_obs, depth_obs, _ = env.observe(attempt_id)
                coord, angle, vis_img = model.predict_grasp(rgb_obs)
                
                camera_height, _ = env.get_camera_target_distance()
                camera_height = camera_height + 0.06
                
                # Save RGB observation
                temp_save_path = os.path.join(vis_dir, f'rgb_obs_{set_num}_{attempt_id}.png')
                imsave(temp_save_path, rgb_obs)
                
                
                target_object = remaining_objects[0]
                print(f"Attempting to pick: {target_object}")
                
                mask = detector.predict_mask(target_object, temp_save_path)
                
                if not mask or not hasattr(mask, 'masks') or len(mask.masks) == 0:
                    print(f"No mask found for {target_object}.")
                    remaining_objects.remove(target_object)
                    continue
                else:
                    # Choose the mask with the largest area
                    best_mask_idx = 0
                    selected_mask = mask.masks[best_mask_idx]
                    binary_mask = selected_mask > 0
                    labeled_mask = label(binary_mask)
                    regions = regionprops(labeled_mask)
                    centroids = [region.centroid for region in regions]
                
                if centroids:
                    # Pick the largest region
                    region_idx = np.argmax([region.area for region in regions])
                    coord = (int(centroids[region_idx][1]), int(centroids[region_idx][0]))
                    angle = predict_grasp_angle_from_numpy(selected_mask)
                    
                    pick_pose = env.image_pose_to_pick_pose(coord, angle, depth_obs - 0.01)
                    result = env.execute_grasp(*pick_pose)
                    
                    if result:
                        env.execute_place()
                    
                    remaining_objects.remove(target_object)
                else:
                    remaining_objects.remove(target_object)
                
                fname = os.path.join(vis_dir, f'{set_num}_{attempt_id}.png')
                imsave(fname, vis_img)
            
            # Calculate objects picked in this set
            final_objects_in_bin = env.num_object_in_tote1()
            objects_picked = n_objects - final_objects_in_bin
            objects_picked_per_set.append(objects_picked)
            print(f"\nSet {set_num + 1}: {objects_picked}/{n_objects} objects picked")

        print(f"\n{'='*50}")
        print("RESULTS")
        print(f"{'='*50}")
        for i, picked in enumerate(objects_picked_per_set):
            print(f"Set {i+1}: {picked}/5 objects picked")
        print(f"\nTotal: {sum(objects_picked_per_set)}/25 objects picked")
        print(f"Average: {sum(objects_picked_per_set)/n_sets:.1f} objects per set")

if __name__ == '__main__':
    main()