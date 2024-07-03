from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
import numpy as np


class ImageNetDataModule(LightningDataModule):
    def __init__(self, data_dir: str, batch_size: int, num_workers: int, train_transforms, val_transforms,
                 pin_memory: bool = True, train_val_test_split=(0.7, 0.15, 0.15)):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.train_val_test_split = train_val_test_split
        self.train_transform = train_transforms
        self.val_transform = val_transforms
        self.imagenet_train = None
        self.imagenet_val = None
        self.imagenet_test = None

    def get_dataset_without_full_tar(self, class_to_exclude='full_tar'):
        dataset = datasets.ImageFolder(self.data_dir, transform=self.val_transform)
        if class_to_exclude in dataset.class_to_idx:
            class_index_to_exclude = dataset.class_to_idx[class_to_exclude]
            filtered_samples = [s for s in dataset.samples if s[1] != class_index_to_exclude]
            old_class_to_idx = {cls: idx for cls, idx in dataset.class_to_idx.items() if
                                idx != class_index_to_exclude}
            new_class_to_idx = {cls: new_idx for new_idx, (cls, old_idx) in
                                enumerate(sorted(old_class_to_idx.items(), key=lambda x: x[1]))}
            updated_samples = [(sample[0], new_class_to_idx[dataset.classes[sample[1]]]) for sample in filtered_samples]

            # Update dataset attributes
            dataset.samples = updated_samples
            dataset.imgs = updated_samples
            dataset.targets = [s[1] for s in updated_samples]
            dataset.class_to_idx = new_class_to_idx
            dataset.classes = [cls for cls in dataset.classes if cls != class_to_exclude]

        return dataset

    def setup(self, stage=None):
        def stratify_split(indices, labels, train_size, val_size, test_size):
            train_indices, temp_indices, _, temp_labels = train_test_split(
                indices, labels, train_size=train_size, stratify=labels)
            relative_val_size = val_size / (val_size + test_size)
            val_indices, test_indices, _, _ = train_test_split(
                temp_indices, temp_labels, train_size=relative_val_size, stratify=temp_labels)
            return train_indices, val_indices, test_indices

        full_dataset = self.get_dataset_without_full_tar()
        indices = list(range(len(full_dataset)))
        labels = np.array([full_dataset.targets[i] for i in indices])

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
