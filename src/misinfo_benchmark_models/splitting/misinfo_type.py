import random
import logging

import duckdb
import datasets

from misinfo_benchmark_models.splitting.utils import subset_dataset_by_article_id

POSITIVE_LABELS = {"Questionable Source", "Conspiracy-Pseudoscience", "Satire"}


def misinfo_type_split_dataset(
    dataset: datasets.Dataset,
    positive_label: str,
    year: int,
    seed: int,
    db_loc: str = "./data/db/misinformation_benchmark_metadata.db",
    val_prop: float = 0.1,
    test_prop: float = 0.2,
):
    metadata_db = duckdb.connect(db_loc, read_only=True)

    assert (
        positive_label == "Questionable Source"
        or positive_label == "Conspiracy-Pseudoscience"
    ), "Positive label must be one of QS or ConPSci"

    logging.info(f"Data - Reserving '{positive_label}' sources for the test set")

    # Find all of the negative (reliable) sources
    negative_labels = metadata_db.sql(
        """
        SELECT DISTINCT label
        FROM sources
        """
    ).fetchall()

    negative_labels = set(map(lambda x: x[0], negative_labels)) - POSITIVE_LABELS

    # Find all of the positive (unreliable) source for the misinfo type we're testing
    test_set_positive_label_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
            SELECT source, label
            FROM sources
            WHERE label = '{positive_label}'
        ) as test_sources ON articles.source = test_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    test_set_positive_label_article_ids = set(
        map(lambda x: x[0], test_set_positive_label_article_ids)
    )

    # Reserve the other unreliable sources for the train set
    train_set_positive_label_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
            SELECT source, label
            FROM sources
            WHERE label IN {tuple(POSITIVE_LABELS - {positive_label})}
        ) as test_sources ON articles.source = test_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    train_set_positive_label_article_ids = set(
        map(lambda x: x[0], train_set_positive_label_article_ids)
    )

    # For each of the negative (reliable) articles, reserve `test_prop`
    # for the test set, stratified over the different reliable label types
    random.seed(seed)

    all_train_set_negative_article_ids = set()
    all_test_set_negative_article_ids = set()
    for negative_label in sorted(list(negative_labels)):
        negative_label_article_ids = metadata_db.sql(
            f"""
            SELECT article_id
            FROM articles INNER JOIN (
                SELECT source, label
                FROM sources
                WHERE label = '{negative_label}'
            ) as train_sources ON articles.source = train_sources.source
            WHERE year = {year}
            """
        ).fetchall()

        negative_label_article_ids = sorted(
            list(map(lambda x: x[0], negative_label_article_ids))
        )

        test_set_negative_label_article_ids = set(
            random.sample(
                population=negative_label_article_ids,
                k=round(test_prop * len(negative_label_article_ids)),
            )
        )

        all_train_set_negative_article_ids.update(
            set(negative_label_article_ids) - test_set_negative_label_article_ids
        )
        all_test_set_negative_article_ids.update(test_set_negative_label_article_ids)

    # Merge the positive and negative articles together
    train_article_ids = (
        all_train_set_negative_article_ids | train_set_positive_label_article_ids
    )

    test_article_ids = (
        all_test_set_negative_article_ids | test_set_positive_label_article_ids
    )

    assert (
        len(train_article_ids & test_article_ids) == 0
    ), "Overlap in train and test article_ids"

    logging.info("Data - Fetched all of the article_ids")

    train_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=train_article_ids
    )
    test_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=test_article_ids
    )

    logging.info("Data - Built train/test datasets")

    actual_val_size = val_prop / (1 - test_prop)

    train_val_dataset = train_dataset.train_test_split(
        test_size=actual_val_size,
        shuffle=True,
        stratify_by_column="label",
        seed=seed,
    )

    logging.info("Data - Built train/val/test datasets")

    dataset_splits = datasets.DatasetDict(
        {
            "train": train_val_dataset["train"],
            "val": train_val_dataset["test"],
            "test": test_dataset,
        }
    )

    return dataset_splits
