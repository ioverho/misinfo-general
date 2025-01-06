import typing
import logging

import pydantic
import omegaconf

logger = logging.getLogger(__name__)


class BaseConfig(pydantic.BaseModel):
    checkpoints_dir: pydantic.DirectoryPath
    data_dir: pydantic.DirectoryPath

    print_args: bool
    dry_run: bool
    debug: bool
    disable_progress_bar: bool

    @classmethod
    def from_hydra(cls, config: omegaconf.DictConfig) -> typing.Self:
        config_dict: typing.Dict[str, typing.Any] = omegaconf.OmegaConf.to_object(
            config
        )

        expected_keys: set = cls.model_fields.keys()

        unused_parameters = config_dict.keys() - expected_keys

        if len(unused_parameters) > 0:
            logger.warning(
                f"When constructing '{cls.__name__}', did not use the following parameters: {set(unused_parameters)}"
            )

        instance: typing.Self = cls(
            **{k: v for k, v in config_dict.items() if k in expected_keys}
        )

        return instance


def log_config(config: pydantic.BaseModel, width: int = 80) -> None:
    config_str: str = config.model_dump_json(
        indent=2,
    )

    logger.info(f"\n{'=' * width}\nPARSED CONFIG:\n\n{config_str}\n{'=' * width}")
