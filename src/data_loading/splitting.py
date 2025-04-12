from typing import Any, Sequence
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset


class SubsetWithLabels(Subset):
    """
    A subset of a dataset where the labels can be overridden with custom ones.

    Attributes:
        new_labels (Sequence[Any]): Custom labels corresponding to the subset indices.
    """

    def __init__(self, dataset: Any, indices: Sequence[int], new_labels: Sequence[Any]):
        """
        Initialize the subset with the dataset, selected indices, and custom labels.

        Args:
            dataset (Any): The full dataset to subset.
            indices (Sequence[int]): Indices to include in the subset.
            new_labels (Sequence[Any]): Labels corresponding to the subset indices.
        """
        super().__init__(dataset, indices)
        self.new_labels = new_labels

    def __getitem__(self, index: int) -> tuple[Any, Any]:
        """
        Retrieve an item and its corresponding custom label.

        Args:
            index (int): Position in the subset.

        Returns:
            tuple: (data, new_label) for the given index.
        """
        data, _ = super().__getitem__(index)
        return data, self.new_labels[index]

    def __getitems__(self, indices: Sequence[int]) -> list[tuple[Any, Any]]:
        """
        Retrieve multiple items and their corresponding custom labels.

        Args:
            indices (Sequence[int]): A list of indices within the subset.

        Returns:
            list: A list of (data, new_label) tuples.
        """
        return [self[i] for i in indices]

    def __len__(self) -> int:
        """
        Return the number of items in the subset.

        Returns:
            int: Length of the subset.
        """
        return len(self.indices)


def stratify_split(
        indices: Sequence[int],
        labels: Sequence[Any],
        train_size: float,
        val_size: float,
        test_size: float
) -> tuple[list[int], list[int], list[int]]:
    """
    Perform a stratified split of the dataset indices into train, validation, and test sets.

    First splits into train and combined val/test sets, then splits the latter into val and test,
    preserving class distribution across all splits.

    Args:
        indices: Dataset sample indices.
        labels: Corresponding labels for stratification.
        train_size: Proportion of data used for training.
        val_size: Proportion of data used for validation.
        test_size: Proportion of data used for testing.

    Returns:
        A tuple of (train_indices, val_indices, test_indices).
    """
    train_indices, eval_indices, _, eval_labels = train_test_split(
        indices,
        labels,
        train_size=train_size,
        stratify=labels
    )

    relative_val_size = val_size / (val_size + test_size)

    val_indices, test_indices, _, _ = train_test_split(
        eval_indices,
        eval_labels,
        train_size=relative_val_size,
        stratify=eval_labels
    )

    return train_indices, val_indices, test_indices