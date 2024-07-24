import logging

import duckdb
import datasets

from misinfo_benchmark_models.splitting.utils import subset_dataset_by_article_id

BIAS_TO_FINE_BIASES = {
    "Left": ["Left", "Extreme Left"],
    "Right": ["Right", "Extreme Right"],
}


def pol_bias_split_dataset(
    dataset: datasets.Dataset,
    positive_bias: str,
    year: int,
    seed: int,
    db_loc: str = "./data/db/misinformation_benchmark_metadata.db",
    val_prop: float = 0.1,
    test_prop: float = 0.2,
):
    assert positive_bias in BIAS_TO_FINE_BIASES, positive_bias

    metadata_db = duckdb.connect(db_loc, read_only=True)

    test_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
                SELECT DISTINCT source
                FROM sources
                WHERE (
                    bias = '{BIAS_TO_FINE_BIASES[positive_bias][0]}'
                    OR bias = '{BIAS_TO_FINE_BIASES[positive_bias][1]}'
                )
            ) AS pol_sources
            ON articles.source = pol_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    train_article_ids = metadata_db.sql(
        f"""
        SELECT *
        FROM articles ANTI JOIN (
                SELECT DISTINCT source
                FROM sources
                WHERE (
                    bias = '{BIAS_TO_FINE_BIASES[positive_bias][0]}'
                    OR bias = '{BIAS_TO_FINE_BIASES[positive_bias][1]}'
                )
            ) AS pol_sources
            ON articles.source = pol_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    test_article_ids = set(map(lambda x: x[0], test_article_ids))
    train_article_ids = set(map(lambda x: x[0], train_article_ids))

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
