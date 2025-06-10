import numpy as np

import matplotlib.pyplot as plt

def error_map_from_file(file1_path, file2_path, output_path):
    # Load the two .npy files
    array1 = np.load(file1_path)
    array2 = np.load(file2_path)
    
    # Calculate percent error
    # Using small epsilon to avoid division by zero
    epsilon = 1e-10
    percent_error = np.abs((array2 - array1) / (array1 + epsilon)) * 100
    
    # Create the plot
    plt.figure(figsize=(10, 8))
    im = plt.imshow(percent_error, cmap='RdYlGn_r')
    plt.clim(0, 150)
    plt.colorbar(im, label='Percent Error (%)')
    
    plt.title('Percent Error Map')
    plt.xlabel('Pixel X')
    plt.ylabel('Pixel Y')
    
    plt.savefig(output_path)

    return percent_error

def error_map_from_arrays(array1, array2):
    # Calculate percent error
    # Using small epsilon to avoid division by zero
    epsilon = 1e-10
    percent_error = np.abs((array2 - array1) / (array1 + epsilon)) * 100
    
    return percent_error
