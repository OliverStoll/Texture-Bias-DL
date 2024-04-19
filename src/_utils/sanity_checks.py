

def print_dataloader_sizes(train_loader, val_loader, test_loader):
    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))
    test_images, test_labels = next(iter(test_loader))
    print(f"Number of batches: {len(train_loader)} | {len(val_loader)} | {len(test_loader)}")
    assert train_images.size() == val_images.size() and train_images.size() == test_images.size()
    assert train_labels.size() == val_labels.size() and train_labels.size() == test_labels.size()
    print(f"Image size: {train_images.size()}\n"
          f"Label size: {train_labels.size()}")