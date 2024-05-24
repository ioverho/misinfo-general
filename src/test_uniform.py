import logging
import sys
from pathlib import Path
import csv

from clearml import Task
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datasets
import duckdb
import hydra
import pandas as pd
import torch
import transformers
import yaml

from misinfo_benchmark_models import SPECIAL_TOKENS
from misinfo_benchmark_models.experiment_metadata import ExperimentMetaData
from misinfo_benchmark_models.labelling import MBFCBinaryLabeller
from misinfo_benchmark_models.data import process_dataset, eval_collator
from misinfo_benchmark_models.splitting import uniform_split_dataset


@hydra.main(version_base="1.3", config_path="../config", config_name="test_uniform")
def test(args: DictConfig):
    if args.year == args.eval_year:
        assert args.fold is not None

    # ==========================================================================
    # Setup
    # ==========================================================================
    # Setup directories for saving output and logs
    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists()
    assert (data_dir / "hf").exists()
    assert (data_dir / "db").exists() or (data_dir / "db_export").exists()

    # Fetch the training configuration =========================================
    model_meta_data = ExperimentMetaData(**args.checkpoint)

    train_task = Task.get_task(
        project_name=model_meta_data.project_name,
        task_name=model_meta_data.task_name,
        allow_archived=False,
        task_filter={
            "type": ["training"],
            "order_by": ["last_update"],
        },
    )

    if train_task is None:
        raise KeyboardInterrupt(
            f"ClearML cannot find train task at: {model_meta_data.project_name} /  {model_meta_data.task_name}"
        )

    checkpoints_dir = (Path(args.checkpoints_dir) / f"{train_task.task_id}").resolve()

    if checkpoints_dir.is_dir():
        checkpoint_dirs = list(checkpoints_dir.glob("checkpoint-*"))

        if len(checkpoint_dirs) > 0:
            checkpoint_loc = sorted(
                checkpoint_dirs, key=lambda x: int(x.stem.split("-")[-1]), reverse=True
            )[0]
        else:
            raise KeyboardInterrupt("No checkpoint directories found!")
    else:
        raise KeyboardInterrupt("ClearML id not found!")

    # Fetches the training configuration
    with open(checkpoints_dir / "config.yaml") as f:
        train_config = yaml.safe_load(f)

    # ClearML logging ==========================================================
    Task.set_random_seed(args.seed)

    task = Task.init(
        project_name=model_meta_data.project_name,
        task_name=f"eval_year[{args.eval_year}]_fold[{args.fold}]",
        task_type="testing",
        tags=[f"train_year[{args.year}]", f"checkpoint[{train_task.task_id}]"],
    )

    task.connect(OmegaConf.to_container(args, resolve=True))
    task.set_parameter(name="model_checkpoint", value=train_task.task_id)

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(str(checkpoints_dir / "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    if args.disable_progress_bar:
        datasets.disable_progress_bars()
        transformers.utils.logging.disable_progress_bar()

    # ==========================================================================
    # Data processing
    # ==========================================================================
    # Build the labeller
    labeller = MBFCBinaryLabeller(data_dir=data_dir)
    logging.info("Data - Built labeller")

    # Load in the tokenizer
    tokenizer = transformers.AutoTokenizer.from_pretrained(train_config["model_name"])
    tokenizer.model_max_length = train_config["data"]["max_length"]

    new_num_tokens = tokenizer.add_tokens(
        list(SPECIAL_TOKENS.values()),
        special_tokens=True,
    )
    logging.info("Data - Fetched tokenizer")
    logging.info(f"Data - Added {new_num_tokens} new tokens")

    dataset = process_dataset(
        data_dir=data_dir,
        year=train_config["year"],
        model_name=train_config["model_name"],
        max_length=train_config["data"]["max_length"],
        batch_size=args.batch_size.tokenization,
        tokenizer=tokenizer,
        labeller=labeller,
        logger=logging,
    )

    # Added steps for eval =====================================================
    # Metadata database
    db_con = duckdb.connect(
        "./data/db/misinformation_benchmark_metadata.db", read_only=True
    )

    year_sources = {
        source
        for (source,) in db_con.sql(
            f"""
            SELECT DISTINCT source
            FROM articles
            WHERE year = {args.year}
            ORDER BY source
            """
        ).fetchall()
    }

    logging.info("Data - Adding an `article_id` column")
    dataset = dataset.sort(column_names=["publication_date", "source"]).map(
        lambda _, idx: {"article_id": f"{args.year:4d}-{idx:07d}"}, with_indices=True
    )

    if args.year == args.eval_year:
        logging.info("Data - Splitting dataset into folds")
        dataset_splits = uniform_split_dataset(
            dataset=dataset,
            seed=train_config["split"]["seed"],
            num_folds=train_config["split"]["num_folds"],
            val_to_test_ratio=train_config["split"]["val_to_test_ratio"],
            cur_fold=args.fold,
        )

        dataset = dataset_splits["test"]
    else:
        logging.info("Data - Filtering dataset")
        dataset = dataset.filter(lambda example: example["source"] in year_sources)

    dataset = dataset.sort(column_names=["article_id"])

    if args.debug:
        logging.info("Data - Verifying metadata and data alignment")
        # Verify metadata db and hf are aligned
        batch_article_ids = pd.DataFrame.from_dict({"article_id": dataset["label"]})  # noqa: F841

        db_con.sql(
            """
            CREATE OR REPLACE TEMP TABLE article_id_checks
            AS SELECT * FROM batch_article_ids
            """
        )

        query = db_con.sql(
            """
            SELECT *
            FROM
                articles INNER JOIN article_id_checks
                ON articles.article_id = article_id_checks.article_id
            ORDER BY article_id_checks.article_id
            """
        )

        i = 0
        while batch := query.fetchmany(2048):
            for db_example in batch:
                hf_example = dataset[i]

                assert hf_example["label"] == db_example[0], (i, db_example, hf_example)
                assert hf_example["source"] == db_example[2], (
                    i,
                    db_example,
                    hf_example,
                )
                assert hf_example["publication_date"] == db_example[5], (
                    i,
                    db_example,
                    hf_example,
                )

                i += 1

    # ==========================================================================
    # Model loading
    # ==========================================================================
    logging.info("Model - Loading model")
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        checkpoint_loc,
        num_labels=len(labeller.int_to_label),
        ignore_mismatched_sizes=True,
    )

    # Add special tokens to the embeddings layer
    model.resize_token_embeddings(
        new_num_tokens=model.get_input_embeddings().weight.shape[0] + new_num_tokens,
        pad_to_multiple_of=64,
    )

    # Freeze all layers of the model except for the embeddings layer
    for p in model.deberta.parameters():
        p.requires_grad = False

    model.eval()

    # ==========================================================================
    # EVAL
    # ==========================================================================

    logging.info("Eval - Running evaluation")

    dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask"],
        output_all_columns=True,
    )

    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.batch_size.eval,
        shuffle=False,
        collate_fn=eval_collator,
    )

    # Overwrite the preds file
    (checkpoints_dir / f"{args.eval_year}_preds.csv").unlink(missing_ok=True)

    for batch in dataloader:
        with torch.inference_mode():
            logits = model.forward(
                input_ids=batch["input_ids"], attention_mask=batch["attention_mask"]
            ).logits

            preds = torch.argmax(input=logits, dim=-1)

        with open(checkpoints_dir / f"{args.eval_year}_preds.csv", "a") as f:
            writer = csv.writer(f)
            for row in zip(batch["article_ids"], preds.tolist(), batch["labels"]):
                writer.writerow(row)

    task.upload_artifact(
        name=f"{args.eval_year}_preds.csv",
        artifact_object=checkpoints_dir / f"{args.eval_year}_preds.csv",
    )


if __name__ == "__main__":
    test()
