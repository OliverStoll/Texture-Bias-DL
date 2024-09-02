import cv2
from torchvision import transforms
from torchvision.transforms import Compose

from grid_shuffle import GridShuffleTransform
from data_init import DataLoaderCollection


def visualize_normalized_image(image, dataset):
    image = image.numpy().transpose(1, 2, 0)
    image = image[:, :, [3, 2, 1]] if dataset == 'bigearthnet' else image
    rescaled_image = ((image + 1) / 2) * 255
    return rescaled_image


def test_grid_shuffle(dataset):
    val_transform = Compose([
        GridShuffleTransform(grid_size=4),
        transforms.Resize(120)
    ])
    dl_collection = DataLoaderCollection()
    dl_tuple = dl_collection.get_dataloader(dataset_name=dataset, model_name='resnet',
                                            is_pretrained=False, val_transform=val_transform)

    train_dl, val_dl, _ = dl_tuple
    train_iterator = iter(train_dl)
    train_imgs, _ = next(train_iterator)
    val_iterator = iter(val_dl)
    val_imgs, _ = next(val_iterator)

    train_images = train_imgs[:5]
    train_images = [visualize_normalized_image(image, dataset) for image in train_images]
    val_images = val_imgs[:5]
    val_images = [visualize_normalized_image(image, dataset) for image in val_images]

    for i in range(5):
        cv2.imwrite(f"../output/{dataset}/{i}_shuffled.jpg", val_images[i])
        cv2.imwrite(f"../output/{dataset}/{i}_original.jpg", train_images[i])


if __name__ == "__main__":
    test_grid_shuffle(dataset='imagenet')
    test_grid_shuffle(dataset='bigearthnet')