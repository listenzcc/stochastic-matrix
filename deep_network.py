"""
File: deep_network.py
Author: Chuncheng Zhang
Date: 2023-10-13
Copyright & Email: chuncheng.zhang@ia.ac.cn

Purpose:
    Amazing things

Functions:
    1. Requirements and constants
    2. Function and class
    3. Play ground
    4. Pending
    5. Pending
"""


# %% ---- 2023-10-13 ------------------------
# Requirements and constants
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights

from rich import print

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'Device is {device}')


# %% ---- 2023-10-13 ------------------------
# Function and class


class ImageTransformer(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ImageTransformer, self).__init__()

        # self.encoder = resnet50(pretrained=True)
        self.encoder = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.encoder.fc = nn.Identity()  # Remove the fully connected layer

        self.transformer = nn.Transformer(
            d_model=input_dim,
            nhead=8,
            num_encoder_layers=6,
            num_decoder_layers=6,
            dim_feedforward=2048,
            dropout=0.1
        )

        self.decoder = nn.Linear(input_dim, output_dim * 100 * 100)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Encode the input image
        encoded_image = self.encoder(x)

        # Reshape the encoded image to match the transformer input shape
        encoded_image = encoded_image.view(
            encoded_image.size(0), -1, encoded_image.size(1))

        # Apply the transformer
        transformed_image = self.transformer(encoded_image, encoded_image)

        # Reshape the transformed image to match the decoder input shape
        print(transformed_image.shape)
        transformed_image = transformed_image.view(
            transformed_image.size(0), transformed_image.size(2))
        print(transformed_image.shape)

        # Apply the decoder
        output = self.decoder(transformed_image)

        output = self.sigmoid(output)
        output = output.view(output.size(0), 3, 100, 100)

        print(output.shape)

        return output


# %% ---- 2023-10-13 ------------------------
# Play ground

# %% ---- 2023-10-13 ------------------------
# Pending
if __name__ == '__main__':
    # Example usage
    input_dim = 2048  # Dimension of the encoded image
    output_dim = 3  # Number of channels in the output image
    batch_size = 2  # 16
    image_size = (6, 6)

    # Create a random input image
    input_image = torch.randn(
        batch_size, 3, image_size[0], image_size[1]).to(device)
    print(input_image.shape)

    # Create the transformer model
    model = ImageTransformer(input_dim, output_dim).to(device)

    # Forward pass
    output_image = model(input_image)

    # 16 x 3 x 5 x 5
    print("Input image shape:", input_image.shape)
    # 16 x 3 x 100 x 100
    print("Output image shape:", output_image.shape)
    print('Done.')

# %% ---- 2023-10-13 ------------------------
# Pending
