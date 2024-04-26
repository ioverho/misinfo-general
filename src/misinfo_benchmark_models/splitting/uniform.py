import numpy as np
from sklearn.model_selection import StratifiedKFold

import datasets


def uniform_split_dataset(
    dataset, seed, num_folds, val_to_test_ratio: float, cur_fold: int
):
    dataset = dataset.shuffle(seed=seed)

    splitter = StratifiedKFold(n_splits=num_folds, shuffle=False)

    train_idx = []
    test_idx = []
    for train_ids, test_ids in splitter.split(
        X=np.empty(dataset.num_rows), y=dataset["label"]
    ):
        train_idx.append(train_ids)
        test_idx.append(test_ids)

    val_idx = test_idx[(cur_fold + 1) % num_folds]
    val_idx = val_idx[: int(val_to_test_ratio * val_idx.shape[0])]

    test_idx = test_idx[cur_fold]

    # train_idx = np.delete(
    #    train_idx[cur_fold], np.searchsorted(train_idx[cur_fold], val_idx)
    # )

    splits = np.full((dataset.num_rows), fill_value="train")

    splits[val_idx] = "val"
    splits[test_idx] = "test"

    dataset = dataset.add_column("split", splits)

    dataset_splits = datasets.DatasetDict(
        {
            "train": dataset.filter(lambda x: x == "train", input_columns="split"),
            "val": dataset.filter(lambda x: x == "val", input_columns="split"),
            "test": dataset.filter(lambda x: x == "test", input_columns="split"),
        }
    )

    dataset_splits = dataset_splits.remove_columns(["split"])

    return dataset_splits
