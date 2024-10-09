from omegaconf import DictConfig, OmegaConf


def print_config(args: DictConfig, logger) -> None:
    logger.info("=" * 80)
    logger.info("CONFIG FILE:\n")
    if args.print_args:
        logger.info(OmegaConf.to_yaml(args, resolve=True))
    else:
        logger.info("Config file not printed.")
    logger.info("=" * 80 + "\n\n")

    if args.dry_run:
        raise KeyboardInterrupt("END OF DRY-RUN")


def save_config(args: DictConfig, results_dir):
    config_yaml = OmegaConf.to_yaml(args, resolve=True)

    (results_dir / "config.yaml").write_text(config_yaml, encoding="utf8")
