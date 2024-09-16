import cv2
import os

from transforms.grid_shuffle import GridShuffleTransform
from transforms.edge_detection import EdgeDetectionTransform
from datasets import DataLoaderCollection


class TransformSanityCheck:
    transforms = {
        'grid_shuffle': GridShuffleTransform(grid_size=4),
        # 'low_pass': LowPassFilterTransform(cutoff=50),  # TODO: implement
        'edges': EdgeDetectionTransform(transform_type='sobel'),
    }
    datasets = ['imagenet', 'bigearthnet']

    def _visualize_image(self, image, dataset):
        image = image.numpy().transpose(1, 2, 0)
        image = image[:, :, [3, 2, 1]] if dataset == 'bigearthnet' else image
        rescaled_image = ((image + 1) / 2) * 255
        return rescaled_image

    def test_transform(self, dataset, transform_name, output_path='../../output/'):
        val_transform = self.transforms[transform_name]
        dl_collection = DataLoaderCollection()
        _, _, transform_dl = dl_collection.get_dataloader(dataset_name=dataset, model_name='resnet', val_transform=val_transform)
        _, _, original_dl = dl_collection.get_dataloader(dataset_name=dataset, model_name='resnet', val_transform=None)
        original_iterator = iter(original_dl)
        original_imgs, _ = next(original_iterator)
        transform_iterator = iter(transform_dl)
        transform_imgs, _ = next(transform_iterator)

        original_images = original_imgs[:5]
        original_images = [self._visualize_image(image, dataset) for image in original_images]
        transform_images = transform_imgs[:5]
        transform_images = [self._visualize_image(image, dataset) for image in transform_images]

        dir_path = f"{output_path}/{transform_name}/{dataset}"
        os.makedirs(dir_path, exist_ok=True)
        for i in range(5):
            cv2.imwrite(f"{dir_path}/{i}_test.jpg", transform_images[i])
            cv2.imwrite(f"{dir_path}/{i}_original.jpg", original_images[i])

    def test_all(self):
        for dataset in self.datasets:
            for transform_name in self.transforms.keys():
                self.test_transform(dataset, transform_name)


if __name__ == "__main__":
    transform_sanity_check = TransformSanityCheck()
    transform_sanity_check.test_transform('bigearthnet', 'edges')