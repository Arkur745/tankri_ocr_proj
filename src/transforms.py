from torchvision.models import ResNet18_Weights
from torchvision import transforms


train_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


val_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])


weights = ResNet18_Weights.DEFAULT



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

train_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    # Geometric augmentations
    transforms.RandomAffine(
        degrees=15,
        translate=(0.10, 0.10),
        scale=(0.90, 1.10),
        shear=8,
        fill=255,
    ),

    transforms.RandomPerspective(
        distortion_scale=0.15,
        p=0.3,
        fill=255,
    ),

    # Blur
    transforms.RandomApply([
        transforms.GaussianBlur(
            kernel_size=3,
            sigma=(0.1, 1.2),
        )
    ], p=0.3),

    # Brightness & Contrast
    transforms.ColorJitter(
        brightness=0.25,
        contrast=0.25,
    ),

    # Convert to Tensor
    transforms.ToTensor(),

    # Random Erasing
    transforms.RandomErasing(
        p=0.15,
        scale=(0.01, 0.05),
        ratio=(0.5, 2.0),
        value=1.0,
    ),

    # Normalize
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