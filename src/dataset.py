from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


class TankriDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        label_to_idx,
        transform=None,
    ):
        self.dataframe = dataframe.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.label_to_idx = label_to_idx
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]

        image_path = self.image_dir / row["image"]

        image = Image.open(image_path).convert("L")

        label = self.label_to_idx[row["label"]]

        if self.transform:
            image = self.transform(image)

        return image, label