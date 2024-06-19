import duckdb
import datasets
from datasets import concatenate_datasets, DatasetDict
from sklearn.model_selection import StratifiedShuffleSplit

from misinfo_benchmark_models.splitting.utils import (
    subset_dataset_by_article_id,
    subset_dataset_by_dataset_id,
)


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

    assert (
        len(covid_article_ids_set & covid_article_ids_anti_set) == 0
    ), "Overlap between datasets."

    dataset = DatasetDict.load_from_disk("../data/hf/")

    full_dataset = concatenate_datasets(
        [dataset[str(year)] for year in [2017, 2018, 2019, 2020, 2021, 2022]]
    )

    train_dataset = subset_dataset_by_article_id(
        dataset=full_dataset, article_ids=covid_article_ids_anti_set
    )

    test_dataset = subset_dataset_by_article_id(
        dataset=full_dataset, article_ids=covid_article_ids_set
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
