import logging

import duckdb
import numpy as np
import datasets

from misinfo_general.splitting.utils import subset_dataset_by_article_id


def topic_split_dataset(
    dataset: datasets.Dataset,
    year: int,
    seed: int,
    db_loc: str = "./data/db/misinformation_benchmark_metadata.db",
    val_prop: float = 0.1,
    test_prop: float = 0.2,
):
    metadata_db = duckdb.connect(db_loc, read_only=True)

    num_articles = len(
        metadata_db.sql(
            f"""
            SELECT *
            FROM articles
            WHERE year = {year}
            """
        )
    )

    # Get sizes for each of the (hyper-)topics
    topic_sizes = metadata_db.sql(
        f"""
        SELECT hyper_topic, count(1) as topic_size
        FROM (
            SELECT topic_id, imbalanced_hyper_cluster as hyper_topic
            FROM topics
            WHERE year = {year}
            ) AS topics
            INNER JOIN articles
            ON topics.topic_id = articles.topic_id
        GROUP BY hyper_topic
        """
    ).fetchall()

    # Get paired sorted lists of hyper_topic size and id
    hyper_topic_ids, hyper_cluster_proportion = list(
        zip(
            *map(
                lambda x: (x[0], x[1] / num_articles),
                sorted(topic_sizes, key=lambda x: x[1]),
            )
        )
    )

    # Select as many hyper_topics as needed to get as close to the desired test set size
    num_of_hyper_topics_for_test_set = (
        np.argmin(np.abs(np.cumsum(hyper_cluster_proportion) - test_prop)) + 1
    )

    logging.info(
        f"Data - Reserving {num_of_hyper_topics_for_test_set} hyper-topics for test set"
    )
    logging.info(
        f"Data - Contains {np.sum(hyper_cluster_proportion[: num_of_hyper_topics_for_test_set])*100:.2f}% of articles"
    )

    hyper_topic_ids_for_test_set = hyper_topic_ids[:num_of_hyper_topics_for_test_set]

    # Get the article_id for the train and test sets
    # Make sure these sets are disjoint
    test_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM (
            SELECT topic_id, imbalanced_hyper_cluster as hyper_topic
            FROM topics
            WHERE year = {year} AND hyper_topic IN {hyper_topic_ids_for_test_set}
            ) AS topics
            INNER JOIN articles
            ON topics.topic_id = articles.topic_id
        """
    ).fetchall()

    train_article_ids = metadata_db.sql(
        f"""
        SELECT article_id
        FROM (
            SELECT topic_id, imbalanced_hyper_cluster as hyper_topic
            FROM topics
            WHERE year = {year} AND hyper_topic NOT IN {hyper_topic_ids_for_test_set}
            ) AS topics
            INNER JOIN articles
            ON topics.topic_id = articles.topic_id
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

    logging.info("Data - Built train/test/val datasets")

    dataset_splits = datasets.DatasetDict(
        {
            "train": train_val_dataset["train"],
            "val": train_val_dataset["test"],
            "test": test_dataset,
        }
    )

    return dataset_splits
