import json

import pandas as pd
import numpy as np
import torch
from data_loading.BENv2DataModule import BENv2DataModule


def get_bigearthnet_dataloader():
    """ Get the dataloaders, using https://git.tu-berlin.de/rsim/BENv2-DataLoading """
    files = {"train": "all_train.csv", "validation": "all_val.csv", "test": "all_test.csv"}
    keys = {}
    for split, file in files.items():
        keys[split] = pd.read_csv(
            f"/media/storagecube/jonasklotz/BigEarthNet-V2/benv2_splits/{file}",
            header=None,
            names=["name"]).name.to_list()

    data_module = BENv2DataModule(
        keys=keys,
        batch_size=32,
        num_workers=30,
        img_size=224,  # the image-size is passed, and will be used as a final resize
        image_lmdb_file="/media/storagecube/jonasklotz/BigEarthNet-V2/BENv2.lmdb",
        label_file="/media/storagecube/jonasklotz/BigEarthNet-V2/lbls.parquet",
        s2s1_mapping_file="/media/storagecube/jonasklotz/BigEarthNet-V2/new_s2s1_mapping.parquet",
        train_transforms=None,
        eval_transforms=None,
        interpolation_mode=None
    )
    data_module.setup()
    return data_module.train_dataloader(), data_module.val_dataloader(), data_module.test_dataloader()





def calculate_class_statistics():
    train_loader, val_loader, test_loader = get_bigearthnet_dataloader()
    print("Train: ", len(train_loader))
    # Initialize a dictionary to hold the count of each class
    positive_counts = None
    idx = 0
    # Iterate through the dataloader
    for batch in train_loader:
        idx += 1
        if idx % 100 == 0:
            print(idx)
        # Assuming batch is a tuple of (inputs, targets)
        inputs, targets = batch

        # Convert targets to a numpy array if it's a tensor
        if isinstance(targets, torch.Tensor):
            targets = targets.numpy()

        # Sum along the batch dimension (axis=0) to get the count of each label in the batch
        batch_label_counts = np.sum(targets, axis=0)

        # Initialize label_counts if it's the first batch
        if positive_counts is None:
            positive_counts = np.zeros_like(batch_label_counts)

        # Accumulate the batch counts
        positive_counts += batch_label_counts

    # Calculate total number of samples
    total_samples = len(train_loader.dataset)
    # calculate the negative samples for each class
    negative_counts = total_samples - positive_counts

    positional_weights = negative_counts / positive_counts

    # Print out the results
    for i, percentage in enumerate(positional_weights):
        print(f"Class {i}: {positional_weights:.1f} ({positive_counts[i]}/{negative_counts[i]})")
        # save a json
        with open("bigearthnet_class_statistics.json", "w") as f:
            json.dump(positional_weights, f)


def calculate_pos_weights():
    file = 'bigearthnet_statistical_analysis.txt'
    # each line is a value for a class
    class_values = []
    with open(file, 'r') as f:
        for line in f:
            class_values.append(round(float(line) * 0.01, 6))
    print(class_values)
    pos_weights = []
    for value in class_values:
        pos_weights.append(round(1 / value, 2))
    print(pos_weights)
    # save to json
    with open("bigearthnet_pos_weights.json", "w") as f:
        json.dump(pos_weights, f)



if __name__ == '__main__':
    calculate_class_statistics()
    # calculate_pos_weights()