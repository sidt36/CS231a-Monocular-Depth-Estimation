import numpy as np
from scipy.spatial.transform import Rotation

def camera_to_world(xpos,
                    xmat,
                    camera_coords):
    camera_coords = np.array(camera_coords)
    xpos = np.array(xpos)
    # Convert rotation matrix to 4x4 homogeneous transform
    T = np.eye(4)
    T[:3, :3] = np.array(xmat).reshape(3, 3)
    T[:3, 3] = xpos

    # Convert camera coordinates to homogeneous coordinates
    camera_coords_h = np.append(camera_coords, 1)

    # Transform to world coordinates
    world_coords = T @ camera_coords_h
    world_coords = world_coords[:3]  # Remove homogeneous component
    
    return world_coords
