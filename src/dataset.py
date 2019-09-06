import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class GesturesDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = pd.read_csv(file_path)
        self.transform = transform

    def __len__(self):
        return len(self.data.index)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]

        # rashape (1, 784) -> (1, 28, 28)
        image = row.loc["pixel1":"pixel784"].values.reshape((1, 28, 28))

        # normalize image
        image = image.astype(np.float32) / 255

        label = row.loc["label"]

        if self.transform:
            image = self.transform(image)

        return image, label
