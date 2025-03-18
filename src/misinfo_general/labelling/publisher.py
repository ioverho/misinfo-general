from misinfo_general.labelling.base import SourceToLabelMapper


class PublisherLabeller(SourceToLabelMapper):
    def __init__(self, data_dir):
        self._labels = set()

        super().__init__(data_dir)

    @property
    def labels(self):
        return sorted(self._labels)

    def map_source_to_label(self, source_tuple):
        publisher_name = source_tuple.source

        self._labels.add(publisher_name)

        return publisher_name
