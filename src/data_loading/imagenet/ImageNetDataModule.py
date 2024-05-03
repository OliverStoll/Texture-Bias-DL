from pytorch_lightning import LightningDataModule, seed_everything
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split, Subset
from sklearn.model_selection import train_test_split
import os
import numpy as np


class ImageNetDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int = 256, num_workers: int = 8,
                 pin_memory: bool = True, train_val_test_split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])  # TODO: make configurable
        self.imagenet_train = None
        self.imagenet_val = None
        self.imagenet_test = None

    def get_dataset(self, class_to_exclude='full_tar'):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.transform)

        # Remove the unwanted class and update dataset structures
        if class_to_exclude in dataset.class_to_idx:
            # Find the index of the class to exclude
            class_index_to_exclude = dataset.class_to_idx[class_to_exclude]

            # Filter out samples belonging to the excluded class
            filtered_samples = [s for s in dataset.samples if s[1] != class_index_to_exclude]

            # Reindex class indices to be contiguous
            old_class_to_idx = {cls: idx for cls, idx in dataset.class_to_idx.items() if
                                idx != class_index_to_exclude}
            new_class_to_idx = {cls: new_idx for new_idx, (cls, old_idx) in
                                enumerate(sorted(old_class_to_idx.items(), key=lambda x: x[1]))}

            # Update the dataset's samples with new class indices
            updated_samples = [(sample[0], new_class_to_idx[dataset.classes[sample[1]]]) for sample in filtered_samples]

            # Update dataset attributes
            dataset.samples = updated_samples
            dataset.imgs = updated_samples  # For compatibility with torchvision versions
            dataset.targets = [s[1] for s in updated_samples]  # Update targets accordingly
            dataset.class_to_idx = new_class_to_idx
            # remove from classes
            dataset.classes = [cls for cls in dataset.classes if cls != class_to_exclude]

        return dataset

    def setup(self, stage=None):
        full_dataset = self.get_dataset()

        # Create indices for full dataset
        indices = list(range(len(full_dataset)))
        labels = np.array([full_dataset.targets[i] for i in indices])

        # Stratify split function
        def stratify_split(indices, labels, train_size, val_size, test_size):
            # Train split
            train_indices, temp_indices, _, temp_labels = train_test_split(
                indices, labels, train_size=train_size, stratify=labels)
            # Adjust val_size relative to remaining dataset
            relative_val_size = val_size / (val_size + test_size)
            # Validation and test split
            val_indices, test_indices, _, _ = train_test_split(
                temp_indices, temp_labels, train_size=relative_val_size, stratify=temp_labels)

            return train_indices, val_indices, test_indices

        # Calculate split sizes
        train_size = self.train_val_test_split[0]
        val_size = self.train_val_test_split[1]
        test_size = self.train_val_test_split[2]

        # Perform stratified split
        train_indices, val_indices, test_indices = stratify_split(indices, labels, train_size,
                                                                  val_size, test_size)

        # Create subsets
        self.imagenet_train = Subset(full_dataset, train_indices)
        self.imagenet_val = Subset(full_dataset, val_indices)
        self.imagenet_test = Subset(full_dataset, test_indices)

    def train_dataloader(self):
        return DataLoader(self.imagenet_train, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def val_dataloader(self):
        return DataLoader(self.imagenet_val, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)

    def test_dataloader(self):
        return DataLoader(self.imagenet_test, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=self.pin_memory)


def exclude_full_tar(path):
    return "full_tar" not in path
