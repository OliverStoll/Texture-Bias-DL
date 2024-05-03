import numpy as np
import torch


def print_dataloader_sizes(train_loader, val_loader, test_loader):
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))
    print(f"Number of batches: {len(train_loader)} | {len(val_loader)} | {len(test_loader)}")
    assert train_images.size() == val_images.size() and train_images.size() == test_images.size()
    assert train_labels.size() == val_labels.size() and train_labels.size() == test_labels.size()
    print(f"Image size: {train_images.size()}\n"
          f"Label size: {train_labels.size()}")


def analyze_class_balance(train_loader, val_loader, test_loader):
    def count_classes(loader):
        class_counts = np.zeros(1000, dtype=int)  # Adjust '100' based on your number of classes
        for _, labels in loader:
            labels_np = labels.numpy()  # Convert labels to NumPy array
            unique, counts = np.unique(labels_np, return_counts=True)
            class_counts[unique] += counts
        return class_counts

    def calculate_proportions(class_counts):
        total = class_counts.sum()
        return class_counts / total if total > 0 else class_counts  # Avoid division by zero

    # Count classes in each DataLoader
    val_counts = count_classes(val_loader)
    train_counts = count_classes(train_loader)
    test_counts = count_classes(test_loader)

    # Calculate proportions
    train_proportions = calculate_proportions(train_counts)
    val_proportions = calculate_proportions(val_counts)
    test_proportions = calculate_proportions(test_counts)

    # Output results
    print("Training data class counts:", train_counts)
    print("Validation data class counts:", val_counts)
    print("Testing data class counts:", test_counts)
    print("Training data class proportions:", train_proportions)
    print("Validation data class proportions:", val_proportions)
    print("Testing data class proportions:", test_proportions)

    return {
        'train_counts': train_counts, 'train_proportions': train_proportions,
        'val_counts': val_counts, 'val_proportions': val_proportions,
        'test_counts': test_counts, 'test_proportions': test_proportions
    }


def sanity_check_dataloader(train_loader, val_loader, test_loader):
    print_dataloader_sizes(train_loader, val_loader, test_loader)
    analyze_class_balance(train_loader, val_loader, test_loader)