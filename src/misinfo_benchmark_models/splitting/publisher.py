from functools import reduce
from itertools import groupby

import duckdb
import numpy as np
import pandas as pd
import datasets
from sklearn.model_selection import StratifiedShuffleSplit


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


def publisher_split_dataset(
    dataset: datasets.Dataset,
    db_loc: str,
    seed: int,
    year: int,
    val_prop: float = 0.2,
    test_prop: float = 0.1,
):
    metadata_db = duckdb.connect(db_loc, read_only=True)

    # Find all sources which have an article in the desired year
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

        select_until_index = np.argmin(np.abs(np.cumsum(group["prop"]) - val_prop)) + 1

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

    # Fetch the article_ids for the train and test splits
    train_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM articles INNER JOIN (
            SELECT source
            FROM sources
            WHERE source NOT IN {tuple(sorted(chosen_test_sources))}
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

    train_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=train_article_ids
    )
    test_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=test_article_ids
    )

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

    return dataset_splits
