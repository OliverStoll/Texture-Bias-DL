from pytorch_lightning import LightningDataModule
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, Subset
from torchvision.transforms import Compose
from sklearn.model_selection import train_test_split
import numpy as np
from common_utils.config import CONFIG
from timm.data import create_transform, resolve_data_config
from timm.data.transforms import ToTensor

from models.models import ModelFactory


DATA_CONFIG = CONFIG['datasets']['imagenet']


class ImageNetDataModule(LightningDataModule):
    ignore_imagenet_transform = ['flexivit']

    def __init__(self,
                 train_transform=None,
                 eval_transform=None,
                 also_use_default_transforms: bool = True,
                 data_dir: str = DATA_CONFIG['path'],
                 batch_size: int = CONFIG['batch_size'],
                 num_workers: int = CONFIG['num_workers'],
                 pin_memory: bool = CONFIG['pin_memory'],
                 train_val_test_split: list[int] = CONFIG['train_val_test_split'],
                 dataloader_timeout: int = CONFIG['dataloader_timeout'],
                 shuffle=False
                 ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.shuffle = shuffle
        self.train_val_test_split = train_val_test_split
        self.train_transform, self.eval_transform = self.get_correct_transforms(
            also_use_default_transforms, train_transform, eval_transform
        )
        self.dataloader_timeout = dataloader_timeout
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_default_transforms(self):
        model = ModelFactory().get_model(
            'resnet',
            dataset_config=CONFIG['datasets']['imagenet'],
            pretrained=False
        )
        config = resolve_data_config({}, model=model)
        return create_transform(**config)

    def get_correct_transforms(self, also_use_default_transforms, input_train_transform, input_eval_transform):
        if not also_use_default_transforms:
            return input_train_transform, input_eval_transform
        base_transform = self._get_default_transforms()
        if input_train_transform is not None:
            combined_train_transform = transforms.Compose([base_transform, input_train_transform])
        else:
            combined_train_transform = base_transform
        if input_eval_transform is not None:
            combined_eval_transform = transforms.Compose([base_transform, input_eval_transform])
        else:
            combined_eval_transform = base_transform
        return combined_train_transform, combined_eval_transform

    def get_dataset_without_full_tar(self, transform, class_to_exclude='full_tar'):
        dataset = datasets.ImageFolder(self.data_dir, transform=transform)
        if class_to_exclude not in dataset.class_to_idx:
            return dataset
        class_index_to_exclude = dataset.class_to_idx[class_to_exclude]
        filtered_samples = [s for s in dataset.samples if s[1] != class_index_to_exclude]
        old_class_to_idx = {cls: idx for cls, idx in dataset.class_to_idx.items() if
                            idx != class_index_to_exclude}
        new_class_to_idx = {cls: new_idx for new_idx, (cls, old_idx) in
                            enumerate(sorted(old_class_to_idx.items(), key=lambda x: x[1]))}
        updated_samples = [(sample[0], new_class_to_idx[dataset.classes[sample[1]]]) for sample
                           in filtered_samples]
        # Update dataset attributes
        dataset.samples = updated_samples
        dataset.imgs = updated_samples
        dataset.targets = [s[1] for s in updated_samples]
        dataset.class_to_idx = new_class_to_idx
        dataset.classes = [cls for cls in dataset.classes if cls != class_to_exclude]
        return dataset

    def _stratify_split(self, indices, labels, train_size, val_size, test_size):
        train_indices, temp_indices, _, temp_labels = train_test_split(
            indices, labels, train_size=train_size, stratify=labels)
        relative_val_size = val_size / (val_size + test_size)
        val_indices, test_indices, _, _ = train_test_split(
            temp_indices, temp_labels, train_size=relative_val_size, stratify=temp_labels)
        return train_indices, val_indices, test_indices

    def setup(self, stage=None):
        full_dataset_train = self.get_dataset_without_full_tar(transform=self.train_transform)
        full_dataset_val = self.get_dataset_without_full_tar(transform=self.eval_transform)
        indices = list(range(len(full_dataset_train)))
        labels = np.array([full_dataset_train.targets[i] for i in indices])

        train_size, val_size, test_size = self.train_val_test_split
        train_indices, val_indices, test_indices = self._stratify_split(
            indices, labels, train_size, val_size, test_size
        )
        # Create subsets
        self.train_dataset = Subset(full_dataset_train, train_indices)
        self.val_dataset = Subset(full_dataset_val, val_indices)
        self.test_dataset = Subset(full_dataset_val, test_indices)

    def get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            timeout=self.dataloader_timeout
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, shuffle=False)
