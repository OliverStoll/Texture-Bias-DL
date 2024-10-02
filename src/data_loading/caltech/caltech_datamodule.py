import torch
from torchvision import transforms
from torch.utils.data import random_split, DataLoader
from torchvision.datasets import Caltech101
import pytorch_lightning as pl
from utils.config import CONFIG

DATA_CONFIG = CONFIG['datasets']['caltech']
to_tensor = transforms.ToTensor()


class CaltechDataModule(pl.LightningDataModule):
    download = False
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    def __init__(
        self,
        train_transforms=None,
        eval_transforms=None,
        also_use_default_transforms: bool = True,  # MINE
        data_dir=DATA_CONFIG['path'],
        batch_size=CONFIG['batch_size'],
        num_workers=CONFIG['num_workers'],
        pin_memory=CONFIG['pin_memory'],
        image_size=DATA_CONFIG['image_size'],
        train_val_test_split=CONFIG['train_val_test_split'],
        dataloader_timeout=CONFIG['dataloader_timeout'],
        seed=CONFIG['seed'],
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.data_dir = data_dir
        self.pin_memory = pin_memory
        self.image_size = image_size
        self.train_val_test_split = train_val_test_split
        self.dataloader_timeout = dataloader_timeout
        self.seed = seed
        self.train_transforms, self.eval_transforms = self.get_correct_transforms(
            also_use_default_transforms, train_transforms, eval_transforms
        )
        # datasets
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def _get_default_transform(self):
        return transforms.Compose([
            transforms.Resize((self.image_size, self.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

    def get_correct_transforms(self, also_use_default_transforms, input_train_transform,
                               input_eval_transform):
        if not also_use_default_transforms:
            return input_train_transform, input_eval_transform
        base_transform = self._get_default_transform()
        if input_train_transform is not None:
            combined_train_transform = transforms.Compose([base_transform, input_train_transform])
        else:
            combined_train_transform = base_transform
        if input_eval_transform is not None:
            combined_eval_transform = transforms.Compose([base_transform, input_eval_transform])
        else:
            combined_eval_transform = base_transform
        return combined_train_transform, combined_eval_transform

    def setup(self, stage=None) -> None:
        full_dataset = Caltech101(
            root=self.data_dir,
            transform=self._get_default_transform(),
            download=self.download
        )
        # test batch
        _test_dl = DataLoader(full_dataset, batch_size=2)
        test_batch = next(iter(_test_dl))

        # Split the dataset into train, val, test
        train_subset, val_subset, test_subset = random_split(
            full_dataset, self.train_val_test_split,
            generator=torch.Generator().manual_seed(self.seed)
        )
        self.train_dataset = torch.utils.data.Subset(
            Caltech101(root=self.data_dir, transform=self.train_transforms), train_subset.indices
        )
        self.val_dataset = torch.utils.data.Subset(
            Caltech101(root=self.data_dir, transform=self.eval_transforms), val_subset.indices
        )
        self.test_dataset = torch.utils.data.Subset(
            Caltech101(root=self.data_dir, transform=self.eval_transforms), test_subset.indices
        )

    def get_dataloader(self, dataset, shuffle):
        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            shuffle=shuffle,
            timeout=self.dataloader_timeout,
            # TODO: fix greyscale images
            # collate_fn=self.collate_fn
        )

    def train_dataloader(self):
        return self.get_dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self.get_dataloader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self.get_dataloader(self.test_dataset, shuffle=False)

    def get_setup_dataloader(self):
        self.setup()
        return self.train_dataloader(), self.val_dataloader(), self.test_dataloader()




if __name__ == '__main__':
    train_dl, val_dl, _ = CaltechDataModule(batch_size=1).get_setup_dataloader()
    print(f"Train batches: {len(val_dl)}")
    iterator = iter(val_dl)
    num_errors = 0
    for idx in range(len(val_dl)):
        try:
            batch = next(iterator)
            print(idx)
        except StopIteration:
            break
        except Exception as e:
            num_errors += 1
            print("ERROR")

    print("ERRORS:", num_errors)

