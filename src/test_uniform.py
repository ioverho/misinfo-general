from itertools import groupby
from pathlib import Path
import csv
import logging
import sys

from accelerate import Accelerator
from clearml import Task
from datasets import concatenate_datasets
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datasets
import duckdb
import hydra
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

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(str(checkpoints_dir / "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )

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

    if args.debug:
        Task.set_offline(offline_mode=True)

    task = Task.init(
        project_name=model_meta_data.project_name,
        task_name=f"eval_year[{args.eval_year}]_fold[{args.fold}]",
        task_type="testing",
        tags=[f"train_year[{args.year}]", f"checkpoint[{train_task.task_id}]"],
        reuse_last_task_id=False,
        continue_last_task=False,
    )

    task.connect(OmegaConf.to_container(args, resolve=True))
    task.set_parameter(name="model_checkpoint", value=train_task.task_id)

    if args.disable_progress_bar:
        datasets.disable_progress_bars()
        transformers.utils.logging.disable_progress_bar()

    logging.info(
        f"Found train checkpoint at {train_task.task_id}: {model_meta_data.project_name}/{model_meta_data.task_name}"
    )

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
        year=args.eval_year,
        model_name=train_config["model_name"],
        max_length=train_config["data"]["max_length"],
        batch_size=args.batch_size.tokenization,
        tokenizer=tokenizer,
        labeller=labeller,
        logger=logging,
    )

    # Added steps for eval =====================================================
    # Metadata database
    metadata_db = duckdb.connect(
        "./data/db/misinformation_benchmark_metadata.db", read_only=True
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
        # Filter out all articles that came from a publisher not in the training set
        logging.info("Data - Filtering dataset")

        # Filter out all articles from publishers not in the training data
        permitted_article_ids = metadata_db.sql(
            f"""
            SELECT article_id
            FROM
                articles INNER JOIN (
                    SELECT DISTINCT source
                    FROM articles
                    WHERE year = {args.year}
                    ) AS year_sources
                ON articles.source = year_sources.source
            WHERE year = {args.eval_year}
            """
        ).fetchall()

        permitted_article_ids = set(map(lambda x: x[0], permitted_article_ids))
        
        #dataset = dataset.filter(
        #    lambda example: example["article_id"] in permitted_article_ids,
        #    num_proc=72,
        #    keep_in_memory=True,
        #)

        # Convert article id to index in the dataset
        article_id_to_dataset_id = {idx: i for i, idx in enumerate(dataset["article_id"])}

        # Keep only the dataset indices which are allowed
        permitted_dataset_ids = sorted(
            [article_id_to_dataset_id[k] for k in article_id_to_dataset_id.keys() & permitted_article_ids]
        )

        # Merge the list of indices into a list of contiguous ranges
        # Ugly, but sooo much fast than a HuggingFace filter/select on Snellius
        out = []
        for _, g in groupby(enumerate(permitted_dataset_ids), lambda k: k[0] - k[1]):
            start = next(g)[1]
            end = list(v for _, v in g) or [start]
            out.append(range(start, end[-1] + 1))

        # Concatenate the sliced datasets together
        dataset = concatenate_datasets([dataset.select(r, keep_in_memory=True) for r in out])

    dataset = dataset.sort(column_names=["article_id"])

    # ==========================================================================
    # Model loading
    # ==========================================================================
    logging.info("Device")

    cuda_available = torch.cuda.is_available()
    if cuda_available:
        logging.info("Device - CUDA available: True")
        logging.info(f"Device - num GPUs: {torch.cuda.device_count()}")
        logging.info(f"Device - current device: {torch.cuda.current_device()}")

    else:
        logging.info("Device - CUDA available: False")
        logging.info("Device - <<< RUNNING ON CPU >>>")

    logging.info("Model - Loading model")

    accelerator = Accelerator(**args.accelerator)

    device = accelerator.device

    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        checkpoint_loc,
        num_labels=len(labeller.int_to_label),
        ignore_mismatched_sizes=True,
    ).to(device)

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

    model, dataloader = accelerator.prepare(
        model, dataloader
    )

    for i, batch in enumerate(dataloader):
        with torch.inference_mode(True):
            logits = model.forward(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits

            preds = torch.argmax(input=logits, dim=-1).cpu()

        with open(checkpoints_dir / f"{args.eval_year}_preds.csv", "a") as f:
            writer = csv.writer(f)
            for row in zip(batch["article_ids"], preds.tolist(), batch["labels"]):
                writer.writerow(row)

        if i == 0 or i % 10 == 0 or i == len(dataloader) - 1:
            logging.info(f"Evaluation - {i} / {len(dataloader)} [{(i / len(dataloader))*100:.2f}%]")

    logging.info("Evaluation - Uploading artifacts")
    task.upload_artifact(
        name=f"{args.eval_year}_preds.csv",
        artifact_object=checkpoints_dir / f"{args.eval_year}_preds.csv",
    )

    logging.info("Finished evaluation")

if __name__ == "__main__":
    test()
