import typing
from dataclasses import dataclass

from omegaconf import OmegaConf


@dataclass(frozen=True, unsafe_hash=True, slots=True)
class ExperimentMetaData:
    generalisation_form: str
    dataset_year: int
    model_name: str
    fold: int
    config: typing.Dict[str, typing.Any]

    @staticmethod
    def convert_to_safe_model_name(name: str):
        return name.replace("-", "_").replace("/", "-")

    @staticmethod
    def convert_to_unsafe_model_name(name: str):
        return name.replace("-", "/").replace("_", "-")

    @property
    def hf_model_name(self):
        return self.convert_to_unsafe_model_name(self.model_name)

    @property
    def loc(self):
        loc = f"{self.generalisation_form}/"
        loc += f"{self.dataset_year}/"
        loc += f"{self.model_name}/"
        loc += f"{self.fold}"

        return loc

    @classmethod
    def from_args(cls, args, generalisation_form: str):
        config = OmegaConf.to_container(args, resolve=True)

        safe_model_name = cls.convert_to_safe_model_name(args.model_name)

        meta_data = cls(
            generalisation_form=generalisation_form,
            dataset_year=str(args.year),
            model_name=safe_model_name,
            fold=str(args.fold),
            config={
                k: v
                for k, v in config.items()
                if k
                in [
                    "year",
                    "seed",
                    "fold",
                    "data",
                    "model",
                    "optim",
                    "model_name",
                ]
            }
            | {"generalisation_form": generalisation_form},
        )

        return meta_data
