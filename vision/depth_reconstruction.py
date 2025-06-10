import numpy as np
import open3d as o3d
import cv2

def create_point_cloud(rgb_image, depth_image, segmentation_mask, camera_intrinsics=None, mask_id=None):
    """
    Create a point cloud from RGB and depth images, filtered by a segmentation mask.
    
    Args:
        rgb_image: RGB image (H, W, 3)
        depth_image: Depth image (H, W) in meters
        segmentation_mask: Segmentation mask (H, W) with different integer values for different objects
        camera_intrinsics: Camera intrinsic parameters as a dictionary with fx, fy, cx, cy keys
                          or None to use default values
        mask_id: ID in the segmentation mask to filter by, or None to use the entire mask
    
    Returns:
        point_cloud: Open3D PointCloud object
    """
    # Default camera intrinsics if not provided (typical for Azure Kinect or similar)
    if camera_intrinsics is None:
        height, width = depth_image.shape
        fx = fy = width / 2  # approximation
        cx, cy = width / 2, height / 2
    else:
        if isinstance(camera_intrinsics, dict):
            fx = camera_intrinsics['fx']
            fy = camera_intrinsics['fy']
            cx = camera_intrinsics['cx']
            cy = camera_intrinsics['cy']
        else:
            # Assuming camera_intrinsics is a 3x3 matrix:
            # [[fx, 0, cx],
            #  [0, fy, cy],
            #  [0,  0,  1]]
            fx = camera_intrinsics[0, 0]
            fy = camera_intrinsics[1, 1]
            cx = camera_intrinsics[0, 2]
            cy = camera_intrinsics[1, 2]
    
    # Create mask based on segmentation
    if mask_id is not None:
        mask = (segmentation_mask == mask_id)
    else:
        mask = (segmentation_mask > 0)  # Use all non-zero segments
    
    # Create mesh grid for pixel coordinates
    v, u = np.indices(depth_image.shape)
    
    # Filter by mask
    u_masked = u[mask]
    v_masked = v[mask]
    z = depth_image[mask]  # depth
    
    # Filter out invalid depth readings
    valid_depth = (z > 0) & (z < 10)  # assume max depth of 10m
    
    # Apply the depth filter to our already masked points
    u_final = u_masked[valid_depth] 
    v_final = v_masked[valid_depth]
    z_final = z[valid_depth]
    
    # Back-project 2D points to 3D
    x = (u_final - cx) * z_final / fx
    y = (v_final - cy) * z_final / fy
    
    # Stack to create points array
    points = np.stack([x, y, z_final], axis=1)
    
    # Check if the RGB image and points align
    if len(points) > 0 and rgb_image.ndim == 3:
        # Make sure indices are within bounds
        v_indices = np.clip(v_final, 0, rgb_image.shape[0] - 1)
        u_indices = np.clip(u_final, 0, rgb_image.shape[1] - 1)
        
        # Get colors from RGB image
        colors = rgb_image[v_indices, u_indices] / 255.0
        
        # Create point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        # Just create a point cloud without colors if no RGB data
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        
        # If no points were found
        if len(points) == 0:
            print("Warning: No valid points found for the point cloud.")
    
    # Return as numpy array for compatibility with your code
    pcd_numpy = np.asarray(pcd.points)
    
    # Print some diagnostics
    print(f"Created point cloud with {len(pcd_numpy)} points")
    if len(pcd_numpy) > 0:
        print(f"Point cloud bounds: min={pcd_numpy.min(axis=0)}, max={pcd_numpy.max(axis=0)}")
    
    return pcd_numpy

def predict_grasp_from_numpy(points):
    """
    Predicts the grasp point for an object by calculating the centroid of its point cloud.
    
    Args:
        points (numpy.ndarray): Nx3 array of point cloud points
    
    Returns:
        tuple: (x, y, z) coordinates of the grasp point
    """
    if len(points) == 0:
        print("Warning: Empty point cloud. Cannot predict grasp point.")
        return (0, 0, 0)
        
    # Calculate the centroid (mean position of all points)
    centroid = np.mean(points, axis=0)
    
    # Extract x, y, z coordinates
    x, y, z = centroid
    
    print(f"Predicted grasp point: ({x:.3f}, {y:.3f}, {z:.3f})")
    return (x, y, z)

def return_grasp_point(rgb_image, depth_image, mask, K):
    """
    Wrapper function to create point cloud and predict grasp point.
    
    Args:
        rgb_image: RGB image (H, W, 3)
        depth_image: Depth image (H, W) in meters
        mask: Segmentation mask (H, W)
        K: Camera intrinsics matrix
    
    Returns:
        tuple: (x, y, z) coordinates of the grasp point
    """
    # Create the point cloud
    try:
        # Print input shapes for debugging
        print(f"RGB shape: {rgb_image.shape}, Depth shape: {depth_image.shape}, Mask shape: {mask.shape}")
        print(f"Camera intrinsics: {K}")
        
        # Make sure mask is binary
        mask_binary = mask > 0
        
        # Print mask statistics
        print(f"Mask statistics: {np.sum(mask_binary)} pixels out of {mask.size} ({100*np.sum(mask_binary)/mask.size:.2f}%)")
        
        # Create the point cloud
        points = create_point_cloud(rgb_image, depth_image, mask_binary, K)
        
        # Predict the grasp point
        grasp_point = predict_grasp_from_numpy(points)
        
        return grasp_point
    except Exception as e:
        import traceback
        print(f"Error in point cloud creation: {e}")
        traceback.print_exc()
        return (0, 0, 0)