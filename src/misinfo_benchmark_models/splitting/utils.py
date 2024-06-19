from itertools import groupby

import datasets


def subset_dataset_by_dataset_id(dataset: datasets.Dataset, dataset_ids: set):
    # Keep only the dataset indices which are allowed
    permitted_dataset_ids = sorted(dataset_ids)

    # Merge the list of indices into a list of contiguous ranges
    # Ugly, but sooo much fast than a HuggingFace filter/select on Snellius
    out = []
    for _, g in groupby(enumerate(permitted_dataset_ids), lambda k: k[0] - k[1]):
        start = next(g)[1]
        end = list(v for _, v in g) or [start]
        out.append(range(start, end[-1] + 1))

    # Concatenate the sliced datasets together
    dataset = datasets.concatenate_datasets(
        [dataset.select(r, keep_in_memory=True) for r in out]
    )

    return dataset


def subset_dataset_by_article_id(dataset: datasets.Dataset, article_ids: set):
    # Convert article id to index in the dataset
    article_id_to_dataset_id = {idx: i for i, idx in enumerate(dataset["article_id"])}

    # Keep only the dataset indices which are allowed
    permitted_dataset_ids = sorted(
        [
            article_id_to_dataset_id[k]
            for k in article_id_to_dataset_id.keys() & article_ids
        ]
    )

    dataset = subset_dataset_by_dataset_id(dataset, permitted_dataset_ids)

    return dataset
