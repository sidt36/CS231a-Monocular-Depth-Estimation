import numpy as np

def predict_grasp_from_numpy(segmented_image, depth_map):
    """
    Predicts the grasp point for an object by calculating the centroid of its point cloud.
    
    Args:
        points (numpy.ndarray): Nx3 array of point cloud points
    
    Returns:
        tuple: (x, y, z) coordinates of the grasp point
    """
    # Calculate the centroid (mean position of all points)
    centroid = np.mean(segmented_image, axis=0)
    
        
    return (x, y, z)


def predict_grasp_angle_from_numpy(segmented_image):
    """
    Predicts the grasp point for an object by calculating the centroid of its point cloud.
    
    Args:
        points (numpy.ndarray): Nx3 array of point cloud points
    
    Returns:
        tuple: (x, y, z) coordinates of the grasp point
    """
    # Calculate the centroid (mean position of all points)
    # Convert segmented image to points for PCA
    y_coords, x_coords = np.where(segmented_image > 0)
    points = np.column_stack((x_coords, y_coords))

    if len(points) > 0:
        centroid = np.mean(points, axis=0)
        
        centered_points = points - centroid
        
        cov_matrix = np.cov(centered_points.T)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        
        # Get minor axis (smallest eigenvalue)
        minor_axis_idx = np.argmin(eigenvalues)
        minor_axis = eigenvectors[:, minor_axis_idx]
        
        # Calculate angle of minor axis
        angle = np.degrees(np.arctan2(minor_axis[1], minor_axis[0]))
    else:
        angle = 0
        
    return angle
if __name__ == "__main__":
    print("test")