import logging
from functools import reduce

import datasets
import duckdb
import numpy as np
import pandas as pd

from .utils import subset_dataset_by_article_id


def limited_publisher_split_dataset(
    dataset: datasets.Dataset,
    db_loc: str,
    year: int,
    seed: int,
    val_prop: float = 0.1,
    test_prop: float = 0.2,
    use_political_bias: bool = False,
    num_sources: int = 1,
):
    metadata_db = duckdb.connect(db_loc, read_only=True)

    label_bias_source_count = metadata_db.sql(
        f"""
            SELECT label, bias, sources.source, count
            FROM (
                SELECT source, count(1) as count
                FROM articles
                WHERE year = {year}
                GROUP BY source
                ORDER BY count DESC
                ) AS year_sources INNER JOIN sources
                ON year_sources.source = sources.source
            ORDER BY label, bias
            """
    ).to_df()

    # Set a convenient index
    label_bias_source_count = label_bias_source_count.set_index(
        keys=["label", "bias"]
    ).sort_index()

    # Select the sources subset for testing for each label-bias combination independently
    # The subset should maximise the number of distinct sources, while being as close to the desired
    # proportion of test articles as possible
    label_bias_selected_sources = dict()
    for label_bias_combo in label_bias_source_count.index.unique().to_numpy():
        group = (
            label_bias_source_count.loc[label_bias_combo[0]]
            .loc[label_bias_combo[1]]
            .copy(deep=True)
        )

        if len(group) <= 1 or isinstance(group, pd.Series):
            continue

        group.loc[:, "prop"] = group["count"] / group["count"].sum()
        group = group.sort_values(by="prop")

        select_until_index = np.argmin(np.abs(np.cumsum(group["prop"]) - test_prop)) + 1

        label_bias_selected_sources[label_bias_combo] = set(
            group.iloc[:select_until_index]["source"]
        )

    # Aggregate all the selected sources together
    chosen_test_sources = reduce(
        lambda a, b: a | b, label_bias_selected_sources.values()
    )

    print(
        f"Reserved {len(chosen_test_sources) / len(label_bias_source_count) * 100:.2f}% of publishers for testing"
    )

    if use_political_bias:
        bias_map = {
            "Extreme Left": "Left",
            "Left": "Left",
            "Left-Center": "Center",
            "Least Biased": "Center",
            "Pro-Science": "Center",
            "Right-Center": "Center",
            "Right": "Right",
            "Extreme Right": "Right",
        }

    else:
        bias_map = {
            "Extreme Left": None,
            "Left": None,
            "Left-Center": None,
            "Least Biased": None,
            "Pro-Science": None,
            "Right-Center": None,
            "Right": None,
            "Extreme Right": None,
        }

    label_bias_source_count_train = label_bias_source_count[
        ~label_bias_source_count["source"].isin(chosen_test_sources)
    ]

    label_bias_source_count_train.index = label_bias_source_count_train.index.map(
        lambda x: (x[0], bias_map.get(x[1], None))
    )

    chosen_train_sources = set()
    for label, bias in label_bias_source_count_train.index.unique():
        label_bias_comb_sources = label_bias_source_count_train.loc[
            (label, bias)
        ].sort_values(by="count", ascending=False)

        train_sources = label_bias_comb_sources.iloc[:num_sources]["source"].to_list()

        chosen_train_sources.update(train_sources)

    # Fetch the article_ids for the train and test splits
    train_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
            SELECT source
            FROM sources
            WHERE source IN {tuple(sorted(chosen_train_sources))}
        ) AS train_sources ON articles.source = train_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    test_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
            SELECT source
            FROM sources
            WHERE source IN {tuple(sorted(chosen_test_sources))}
        ) AS test_sources ON articles.source = test_sources.source
        WHERE year = {year}
        """
    ).fetchall()

    train_article_ids = set(map(lambda x: x[0], train_article_ids))
    test_article_ids = set(map(lambda x: x[0], test_article_ids))

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
