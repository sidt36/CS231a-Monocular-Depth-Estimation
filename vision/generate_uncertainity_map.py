import torch
import torch.nn as nn
import numpy as np

class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1,
                               kernel_size=3,
                               stride=2,
                               padding=1,
                               output_padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class UncertaintyAutoencoder:
    def __init__(self, weights_path=r'C:\CS231A-Project\vision\models\depth_error_prediction_normalized_autoencoder.pth', device=None):
        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.model = Autoencoder()
        self.model.load_state_dict(torch.load(weights_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()

    def predict(self, rgb_img, depth_img):
        # Normalize and stack
        if rgb_img.max() > 1.0:
            rgb_img = rgb_img / 255.0
        if depth_img.max() > 1.0:
            depth_img = depth_img / 255.0

        rgbd = np.concatenate([rgb_img, depth_img[..., None]], axis=-1)
        rgbd = np.transpose(rgbd, (2, 0, 1))
        rgbd_tensor = torch.from_numpy(rgbd).float().unsqueeze(0).to(self.device)  # (1, 4, H, W)

        with torch.no_grad():
            output = self.model(rgbd_tensor)
        return output.cpu()
