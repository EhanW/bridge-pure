from PIL import Image
import os
import numpy as np
from torch.utils.data import Dataset

class NPZDataset(Dataset):
    def __init__(self, root, train, transform=None, data_path=None):
        """
        Args:
            npz_file (string): Path to the npz file.
            transform (callable, optional): Optional transform to be applied
                on a sample (such as image normalization, augmentation, etc.).
        """
        if data_path is not None:
            self.data = np.load(data_path)['data']
            self.targets = np.load(data_path)['targets']
        else:
            if train:
                npz_file = os.path.join(root, 'train_dataset.npz')
            else:
                npz_file = os.path.join(root, 'test_dataset.npz')
            self.dataset = np.load(npz_file)
            self.data = self.dataset['data']  # Assumes the npz file has 'data' key for images
            self.targets = self.dataset['targets']  # Assumes the npz file has 'targets' key for labels
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.targets[idx]
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image, label
    