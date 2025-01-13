from itertools import product
from collections import Counter
import warnings
import typing
import logging

import duckdb
import numpy as np
import pandas as pd
import datasets
from misinfo_general.splitting.utils import subset_dataset_by_article_id

LABEL_MAPPING = {
    "Left": "Reliable",
    "Left-Center": "Reliable",
    "Least Biased": "Reliable",
    "Right-Center": "Reliable",
    "Right": "Reliable",
    "Pro-Science": "Reliable",
    "Satire": None,
    "Questionable Source": "Questionable Source",
    "Conspiracy-Pseudoscience": "Conspiracy-Pseudoscience",
}


def dataset_map_split_dataset(
    dataset: datasets.Dataset,
    year: int,
    seed: int,
    db_loc: str = "./data/db/misinformation_benchmark_metadata.db",
    val_prop: float = 0.1,
    test_prop: float = 0.2,
    split: int = 0,
    num_buckets: int = 15,
    publisher_occurences: int = 5,
) -> datasets.DatasetDict:
    def reserve_publishers_for_testing(
        group_idx: tuple[str, str],
        publisher_occurences: int,
        num_buckets: int,
        seed: int,
        min_publisher_counts: int = 50,
    ) -> typing.List[set]:
        label_bias_source_count_ = label_bias_source_count.set_index(
            keys=["coarse_labelling", "bias"]
        )

        with warnings.catch_warnings():
            warnings.simplefilter(
                action="ignore", category=pd.errors.PerformanceWarning
            )
            group = label_bias_source_count_.loc[group_idx]

        min_count = min(
            min_publisher_counts, np.quantile(group["count"].to_numpy(), q=[0.1])[0]
        )

        group_filtered = group[group["count"] >= min_count]

        eligible_publishers = sorted(list(group_filtered["source"]))

        combinations_counter = Counter()

        buckets = [(m, set()) for m in range(num_buckets)]

        rng = np.random.default_rng(seed=seed)

        for _ in range(publisher_occurences):
            eligible_publishers = rng.permuted(eligible_publishers)

            for publisher in eligible_publishers:
                eligible_buckets = filter(lambda x: publisher not in x[1], buckets)

                scored_eligible_buckets = []
                for bucket_candidate in eligible_buckets:
                    bucket_candidate_score = sum(
                        combinations_counter.get((publisher, publisher_b), 0) ** 10
                        for publisher_b in bucket_candidate[1]
                    )

                    scored_eligible_buckets.append(
                        (
                            bucket_candidate[0],
                            bucket_candidate[1],
                            bucket_candidate_score,
                        )
                    )

                _, _, min_score = min(scored_eligible_buckets, key=lambda x: x[2])

                eligible_buckets = list(
                    filter(lambda x: x[2] == min_score, scored_eligible_buckets)
                )

                chosen_bucket_idx, _, _ = min(eligible_buckets, key=lambda x: len(x[1]))

                for combo in list(
                    product([publisher], list(buckets[chosen_bucket_idx][1]))
                ):
                    combinations_counter[tuple(sorted(combo))] += 1

                buckets[chosen_bucket_idx][1].add(publisher)

        buckets = list(map(lambda x: x[1], buckets))

        return buckets

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

    label_bias_source_count[~label_bias_source_count["bias"].notna()]["bias"] = (
        label_bias_source_count[
            ~label_bias_source_count["bias"].notna()
        ]["bias"].replace(np.nan, None)
    )

    label_bias_source_count["coarse_labelling"] = label_bias_source_count.label.map(
        lambda x: LABEL_MAPPING[x]
    )

    label_bias_combos = list(
        map(
            tuple,
            label_bias_source_count[["coarse_labelling", "bias"]].to_records(
                index=False
            ),
        )
    )

    label_bias_combos = Counter(label_bias_combos)

    chosen_test_sources = set()
    for label_bias_combo in label_bias_combos.keys():
        buckets = reserve_publishers_for_testing(
            group_idx=label_bias_combo,
            num_buckets=num_buckets,
            publisher_occurences=publisher_occurences,
            seed=seed,
        )

        split_bucket = buckets[split]

        chosen_test_sources.update(split_bucket)

    logging.info(f"Data - Reserved {len(chosen_test_sources)} publishers for testing")

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

    # Split the dataset into train and test
    train_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=train_article_ids
    )
    test_dataset = subset_dataset_by_article_id(
        dataset=dataset, article_ids=test_article_ids
    )

    # Figure out how much of the training dataset to reserve for validation
    test_prop = test_dataset.num_rows / (train_dataset.num_rows + test_dataset.num_rows)
    actual_val_size = val_prop / (1 - test_prop)

    # Split the dataset
    train_val_dataset = train_dataset.train_test_split(
        test_size=actual_val_size,
        shuffle=True,
        stratify_by_column="label",
        seed=seed,
    )

    dataset_splits = datasets.DatasetDict(
        {
            "train": train_val_dataset["train"],
            "val": train_val_dataset["test"],
            "test": test_dataset,
        }
    )

    return dataset_splits
