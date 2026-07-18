from torchvision.models import ResNet18_Weights
from torchvision import transforms
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from PIL import Image, ImageFilter
import random
import io
import numpy as np
import torch

train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


weights = ResNet18_Weights.DEFAULT


"""Baseline model transformations"""
# train_transform_resnet = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),

#     transforms.RandomAffine(
#         degrees=20,
#         translate=(0.15, 0.15),
#         scale=(0.8, 1.2),
#         shear=15,
#         fill=255,
#     ),

#     transforms.ToTensor(),

#     transforms.Normalize(
#         mean=[0.485, 0.456, 0.406],
#         std=[0.229, 0.224, 0.225],
#     ),
# ])

# val_transform_resnet = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=weights.transforms().mean,
#         std=weights.transforms().std,
#     ),
# ])

"""AugOCR Experimentation"""
# train_transform_resnet = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),

#     # Geometric augmentations
#     transforms.RandomAffine(
#         degrees=15,
#         translate=(0.10, 0.10),
#         scale=(0.90, 1.10),
#         shear=8,
#         fill=255,
#     ),

#     transforms.RandomPerspective(
#         distortion_scale=0.15,
#         p=0.3,
#         fill=255,
#     ),

#     # Blur
#     transforms.RandomApply([
#         transforms.GaussianBlur(
#             kernel_size=3,
#             sigma=(0.1, 1.2),
#         )
#     ], p=0.3),

#     # Brightness & Contrast
#     transforms.ColorJitter(
#         brightness=0.25,
#         contrast=0.25,
#     ),

#     # Convert to Tensor
#     transforms.ToTensor(),

#     # Random Erasing
#     transforms.RandomErasing(
#         p=0.15,
#         scale=(0.01, 0.05),
#         ratio=(0.5, 2.0),
#         value=1.0,
#     ),

#     # Normalize
#     transforms.Normalize(
#         mean=weights.transforms().mean,
#         std=weights.transforms().std,
#     ),
# ])

# val_transform_resnet = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.Grayscale(num_output_channels=3),
#     transforms.ToTensor(),
#     transforms.Normalize(
#         mean=weights.transforms().mean,
#         std=weights.transforms().std,
#     ),
# ])

"""Domain Specific augmentation"""

weights = ResNet18_Weights.DEFAULT


class MotionBlur:
    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        # Approximate motion blur using BoxBlur
        radius = random.uniform(0.5, 1.5)
        return img.filter(ImageFilter.BoxBlur(radius))


class JPEGCompression:
    def __init__(self, p=0.30, quality=(35, 90)):
        self.p = p
        self.quality = quality

    def __call__(self, img):
        if random.random() > self.p:
            return img

        buffer = io.BytesIO()
        q = random.randint(*self.quality)
        img.save(buffer, format="JPEG", quality=q)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")


class AddGaussianNoise:
    def __init__(self, std=0.02, p=0.25):
        self.std = std
        self.p = p

    def __call__(self, tensor):
        if random.random() > self.p:
            return tensor

        noise = torch.randn_like(tensor) * self.std
        return torch.clamp(tensor + noise, 0.0, 1.0)


train_transform_resnet = transforms.Compose([

    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    # Existing geometric augmentations
    transforms.RandomAffine(
        degrees=15,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10),
        shear=8,
        fill=255,
    ),

    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.30,
        fill=255,
    ),

    # Existing blur
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.2),
        )
    ], p=0.30),

    # NEW Motion Blur
    MotionBlur(p=0.15),

    # Existing brightness/contrast
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
    ),

    # NEW JPEG Compression
    JPEGCompression(
        p=0.30,
        quality=(35, 90),
    ),

    transforms.ToTensor(),

    # NEW Sensor Noise
    AddGaussianNoise(
        std=0.02,
        p=0.25,
    ),

    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])

val_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=weights.transforms().mean,
        std=weights.transforms().std,
    ),
])
