import numpy as np
import torch
from common_utils.logger import create_logger
from common_utils.config import CONFIG


log = create_logger("SanityCheck Data")


def print_dataloader_sizes(train_loader, val_loader, test_loader):
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))
    assert train_images.size() == val_images.size() and train_images.size() == test_images.size()
    assert train_labels.size() == val_labels.size() and train_labels.size() == test_labels.size()
    log.debug(f"Batches, Images, Labels: "
              f"[{len(train_loader)}, {len(val_loader)}, {len(test_loader)}]"
              f" | {list(train_images.size())}"
              f" | {list(train_labels.size())}")


def check_image_normalization(train_loader, val_loader, test_loader):
    train_images, _ = next(iter(train_loader))
    val_images, _ = next(iter(val_loader))
    test_images, _ = next(iter(test_loader))
    log.debug(f"Normalization: [{round(float(train_images.max()), 1)},{round(float(train_images.min()), 1)}]"
              f" | [{round(float(val_images.max()), 1)},{round(float(val_images.min()), 1)}]"
              f" | [{round(float(test_images.max()), 1)},{round(float(test_images.min()), 1)}]")


def analyze_dataloader_class_balance(train_loader, val_loader, test_loader):
    def count_classes(loader):
        if isinstance(loader.dataset, torch.utils.data.Subset):
            num_classes = 1000  # imagenet
        else:
            num_classes = 19  # bigearthnet
        class_counts = np.zeros(num_classes, dtype=int)
        for _, labels in loader:
            labels_np = labels.numpy()  # Convert labels to NumPy array
            unique, counts = np.unique(labels_np, return_counts=True)
            class_counts[unique] += counts
        return class_counts

    def calculate_proportions(class_counts):
        total = class_counts.sum()
        return class_counts / total if total > 0 else class_counts  # Avoid division by zero

    log.debug("Starting class balance analysis...")

    # Count classes in each DataLoader
    val_counts = count_classes(val_loader)
    train_counts = count_classes(train_loader)
    test_counts = count_classes(test_loader)

    # Calculate proportions
    train_proportions = calculate_proportions(train_counts)
    val_proportions = calculate_proportions(val_counts)
    test_proportions = calculate_proportions(test_counts)

    # Output results
    log.debug(f"Data class proportions:\n {np.round(train_proportions, 4)}\n\n\n"
              f"{np.round(val_proportions, 4)}\n\n\n{np.round(test_proportions, 4)}")
    log.debug(f"Data class Max-Min:"
              f"\n[{np.round(train_proportions.min(), 4)},{np.round(train_proportions.max(), 4)}]"
              f"\n\n\n[{np.round(val_proportions.min(), 4)},{np.round(val_proportions.max(), 4)}]"
              f"\n\n\n[{np.round(test_proportions.min(), 4)},{np.round(test_proportions.max(), 4)}]")

    return {
        'train_counts': train_counts, 'train_proportions': train_proportions,
        'val_counts': val_counts, 'val_proportions': val_proportions,
        'test_counts': test_counts, 'test_proportions': test_proportions
    }


def check_dataloader(train_loader, val_loader, test_loader):
    print_dataloader_sizes(train_loader, val_loader, test_loader)
    check_image_normalization(train_loader, val_loader, test_loader)


def check_all(train_loader, val_loader, test_loader):
    check_dataloader(train_loader, val_loader, test_loader)
    analyze_dataloader_class_balance(train_loader, val_loader, test_loader)