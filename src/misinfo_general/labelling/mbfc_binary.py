from .base import SourceToLabelMapper


class MBFCBinaryLabeller(SourceToLabelMapper):
    mbfc_label_condensed = {
        "Left": "reliable",
        "Left-Center": "reliable",
        "Least Biased": "reliable",
        "Right-Center": "reliable",
        "Right": "reliable",
        "Pro-Science": "reliable",
        "Satire": "unreliable",
        "Questionable Source": "unreliable",
        "Conspiracy-Pseudoscience": "unreliable",
    }

    labels = ["reliable", "unreliable"]

    def map_source_to_label(self, source_tuple):
        return self.mbfc_label_condensed[source_tuple.label]
