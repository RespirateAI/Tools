import os
import numpy as np
import random
from torch.utils.data import Dataset


def custom_collate(batch):
    slide_names = [item["slide_name"] for item in batch]
    labels = [item["label"] for item in batch]
    features = [item["features"] for item in batch]

    return {"slide_name": slide_names, "features": features, "label": labels}


class LungDataset(Dataset):
    def __init__(self, destination_directory):
        self.destination_directory = destination_directory
        self.file_list = [
            filename
            for filename in os.listdir(destination_directory)
            if filename.endswith(".npy")
        ]
        random.shuffle(self.file_list)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_name = self.file_list[idx]
        file_path = os.path.join(self.destination_directory, file_name)

        slide_name = file_name[:-4]  # Slide name taken from file name
        data = np.load(file_path, allow_pickle=True)
        # print(f"Size in dataloader {data.shape}")
        features = (
            data.tolist()
        )  # Convert to list because Variable size tensors cannot be concatenated

        if slide_name[8] == "A":
            label = 0
        elif slide_name[8] == "G":
            label = 1

        return {"slide_name": slide_name, "features": features, "label": label}
