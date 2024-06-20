from itertools import groupby
import logging

import duckdb
import datasets
from sklearn.model_selection import StratifiedShuffleSplit


def subset_dataset_by_dataset_id(dataset: datasets.Dataset, dataset_ids: set):
    # Keep only the dataset indices which are allowed
    permitted_dataset_ids = sorted(dataset_ids)

    # Merge the list of indices into a list of contiguous ranges
    # Ugly, but sooo much faster than a HuggingFace filter/select on Snellius
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
    dataset = dataset.filter(
        lambda x: x["article_id"] in article_ids,
        keep_in_memory=True,
        num_proc=17,
    )

    return dataset


def covid_split_dataset(
    dataset: datasets.Dataset,
    db_loc: str,
    seed: int,
    val_prop: float = 0.2,
    test_prop: float = 0.1,
):
    metadata_db = duckdb.connect(db_loc, read_only=True)

    covid_article_ids_set = set(
        map(
            lambda x: x[0],
            metadata_db.sql(
                """
                SELECT article_id
                FROM covid_articles
                """
            ).fetchall(),
        )
    )

    covid_article_ids_anti_set = set(
        map(
            lambda x: x[0],
            metadata_db.sql(
                """
                SELECT articles.article_id
                FROM articles
                    ANTI JOIN covid_articles
                    ON covid_articles.article_id = articles.article_id
                """
            ).fetchall(),
        )
    )

    logging.info("Data - Fetched article_ids")

    assert (
        len(covid_article_ids_set & covid_article_ids_anti_set) == 0
    ), "Overlap between dataset `article_id`."

    train_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=covid_article_ids_anti_set
    )

    test_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=covid_article_ids_set
    )

    logging.info("Data - Built train/test datasets")

    actual_test_size = test_prop / (1 - val_prop)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=(1 - actual_test_size),
        test_size=actual_test_size,
        random_state=seed,
    )

    train_idx, val_idx = next(
        splitter.split(X=train_dataset["label"], y=train_dataset["label"])
    )

    dataset_splits = datasets.DatasetDict(
        {
            "train": subset_dataset_by_dataset_id(train_dataset, train_idx),
            "val": subset_dataset_by_dataset_id(train_dataset, val_idx),
            "test": test_dataset,
        }
    )

    logging.info("Data - Built train/test/val datasets")

    return dataset_splits
