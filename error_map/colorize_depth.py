import numpy as np
import matplotlib.pyplot as plt
import cv2

def colorize_depth(depth_path, output_path, title, normalize=True, invert = True):
    # Read depth image
    depth_img = np.load(depth_path)  # Assuming the depth image is saved in .npy format
    depth_img = depth_img.astype(np.float32)

    if normalize:
        # Normalize the depth image to the range [0, 1]
        depth_img = (depth_img - depth_img.min()) / (depth_img.max() - depth_img.min())

    if invert:
        depth_img = 1 - depth_img  # Invert the depth image

    colormap = plt.colormaps.get_cmap('plasma') # You can choose other colormaps like 'plasma', 'magma', 'inferno', 'coolwarm'
    colored_depth_image = colormap(depth_img)
    
    # Display the result
    plt.imshow(colored_depth_image)
    plt.colorbar(label='Depth').ax.invert_yaxis()
    plt.title(title)
    plt.show()
    
    plt.imsave(output_path, colored_depth_image)
    np.save('normalized' + output_path.replace('.png', '.npy'), depth_img)  # Save the colorized depth image as .npy file
    return colored_depth_image

