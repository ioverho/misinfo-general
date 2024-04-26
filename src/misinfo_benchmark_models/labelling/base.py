from abc import ABC, abstractmethod

import duckdb


class SourceToLabelMapper(ABC):
    def __init__(self, metadata_db_loc) -> None:
        metadata_db = duckdb.connect(str(metadata_db_loc))

        self.source_to_label = dict()
        for source_tuple in self.fetch_sources(metadata_db=metadata_db):
            self.source_to_label[source_tuple.source] = self.map_source_to_label(
                source_tuple
            )

        self.label_to_int = dict()
        for i, label in enumerate(self.int_to_label):
            self.label_to_int[label] = i

        metadata_db.close()

    def fetch_sources(self, metadata_db):
        source_tuples = (
            metadata_db.sql(
                """
                SELECT *
                FROM sources
                """
            )
            .df()
            .itertuples()
        )

        return source_tuples

    @abstractmethod
    def map_source_to_label(self, source_tuple):
        raise NotImplementedError

    @property
    @abstractmethod
    def labels(self):
        raise NotImplementedError

    @property
    def int_to_label(self):
        return self.labels

    @property
    def source_to_int(self):
        return {k: self.label_to_int[v] for k, v in self.source_to_label.items()}

    def __call__(self, source: str, return_int: bool = False):
        label = self.source_to_label[source]

        if return_int:
            return self.label_to_int[label]
        else:
            return label
