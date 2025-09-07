import torch
from torch.utils.data import Dataset, DataLoader


class VaeDataset(Dataset):
    def __init__(self):
        super(VaeDataset, self).__init__()
        self.train_images_path = "data/train_images"
        self.val_images_path = "data/val_images"



