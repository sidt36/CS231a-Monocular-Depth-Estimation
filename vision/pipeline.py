from segmentation_utils import get_green_mask
from depth_reconstruction import create_point_cloud, predict_grasp_from_numpy
 

import cv2
import numpy as np

import matplotlib.pyplot as plt


def return_grasp_point(rgb_image, depth_image, K):
    mask = get_green_mask(rgb_image)
    pcd_base = create_point_cloud(rgb_image, depth_image, mask, K)
    grasp_location = predict_grasp_from_numpy(pcd_base)

    return grasp_location

if __name__ == "__main__":
    print("test")