import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from colorize_depth import colorize_depth
from error_map import error_map_from_file
import DepthAnythingV2


DEVICE = 'cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu'

model_configs = {
    'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'vitg': {'encoder': 'vitg', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]}
}

encoder = 'vitl' # or 'vits', 'vitb', 'vitg'

model = DepthAnythingV2(**model_configs[encoder])
model.load_state_dict(torch.load(f'checkpoints/depth_anything_v2_{encoder}.pth', map_location='cpu'))
model = model.to(DEVICE).eval()

raw_img = cv2.imread('ee_rgb_0.png') # HxWx3 BGR image
raw_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB) # HxWx3 RGB image
depth = model.infer_image(raw_img) # HxW raw depth map in numpy
depth = depth.astype('float32') # HxW float32 depth map
np.save('ee_DA_0.npy', depth) # Save the depth map as a .npy file

colored_depth_image = colorize_depth("ee_depth_0.npy", "colorized_ee_depth_0.png", "Normalized Depth Image", True, False)
colored_DA_image = colorize_depth("ee_DA_0.npy", "colorized_ee_DA_0.png", "Normalized Monocular Depth Estimation Image", True, True)

errormap = error_map_from_file("normalizedcolorized_ee_depth_0.npy", "normalizedcolorized_ee_DA_0.npy")

# plt.imshow(raw_img)
# plt.title('RGB Virtual Image')
# plt.show()

fig, axs = plt.subplots(2, 2, figsize=(10, 10))
im1 = axs[0, 0].imshow(raw_img)
axs[0, 0].set_title('RGB Virtual Image')
im1.fig_size = (5,5)

error_opts = {'vmin': 0, 'vmax': 150, 'cmap': 'RdYlGn_r'}
im2 = axs[0, 1].pcolormesh(errormap, **error_opts)
axs[0, 1].set_title('Percent Error Map')
axs[0, 1].invert_yaxis()
plt.colorbar(im2, ax=axs[0, 1], label='Percent Error (%)')
im2.fig_size = (5,5)

im3 = axs[1, 0].imshow(colored_depth_image)
plt.colorbar(im3, ax=axs[1, 0]).ax.invert_yaxis()
axs[1, 0].set_title('Normalized Depth Image')
im3.fig_size = (5,5)

im4 = axs[1, 1].imshow(colored_DA_image)
plt.colorbar(im4, ax=axs[1, 1]).ax.invert_yaxis()
axs[1, 1].set_title('Normalized Monocular Depth Estimation Image')
im4.fig_size = (5,5)

plt.tight_layout()
plt.show()




