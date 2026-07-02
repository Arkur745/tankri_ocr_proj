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

train_transform_resnet = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),

    transforms.RandomAffine(
        degrees=20,
        translate=(0.15, 0.15),
        scale=(0.8, 1.2),
        shear=15,
        fill=255,
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
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
