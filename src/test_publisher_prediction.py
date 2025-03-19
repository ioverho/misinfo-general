from pathlib import Path
import csv
import logging
import sys

from accelerate import Accelerator
from clearml import Task
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
import datasets
import hydra
import torch
import torch.nn.functional as F
import transformers
import yaml

from misinfo_general import SPECIAL_TOKENS
from misinfo_general.labelling import PublisherLabeller
from misinfo_general.data import process_dataset, eval_collator
from misinfo_general.splitting import uniform_split_dataset
from misinfo_general.utils import print_config


@hydra.main(version_base="1.3", config_path="../config", config_name="test_dataset_map")
def test(args: DictConfig):
    assert args.year is not None
    assert args.checkpoint.model_name is not None

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            # logging.FileHandler(str(checkpoints_dir / "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Print config for logging purposes
    print_config(args, logger=logging)

    # ==========================================================================
    # Setup
    # ==========================================================================
    # Setup directories for saving output and logs
    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists()
    assert (data_dir / "hf").exists()
    assert (data_dir / "db").exists() or (data_dir / "db_export").exists()

    # Fetch the training configuration =========================================
    project_name = "misinfo_benchmark_models/publisher_prediction"
    task_name = f"year[{args.checkpoint.year}]_model[{args.checkpoint.model_name}]_fold[{args.checkpoint.fold}]"

    logging.info(
        f"Looking for train task in: {project_name}/{task_name}"
    )

    # TODO: make dynamic
    train_task = Task.get_task(
        project_name=project_name,
        task_name=task_name,
    )

    if train_task is None:
        raise KeyboardInterrupt(
            f"ClearML cannot find train task at: {project_name}/{task_name}"
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

    # TODO: make dynamic
    task = Task.init(
        project_name=project_name,
        task_name=f"year[{args.checkpoint.year}]_fold[{args.fold}]",
        task_type="testing",
        tags=[
            f"train_year[{args.checkpoint.year}]",
            f"checkpoint[{train_task.task_id}]",
            f"split[{args.split.split_name}]",
        ],
        reuse_last_task_id=False,
        continue_last_task=False,
    )

    task.connect(OmegaConf.to_container(args, resolve=True))
    task.set_parameter(name="model_checkpoint", value=train_task.task_id)

    if args.disable_progress_bar:
        datasets.disable_progress_bars()
        transformers.utils.logging.disable_progress_bar()

    logging.info(
        f"Found train checkpoint at {train_task.task_id}: {project_name}/{task_name}"
    )

    # ==========================================================================
    # Data processing
    # ==========================================================================
    # Build the labeller
    labeller = PublisherLabeller(data_dir=data_dir)
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

    # Process the data
    dataset = process_dataset(
        data_dir=data_dir,
        year=args.year,
        model_name=train_config["model_name"],
        max_length=train_config["data"]["max_length"],
        batch_size=args.batch_size.tokenization,
        tokenizer=tokenizer,
        labeller=labeller,
        logger=logging,
        dataset_tag="publisher_prediction",
    )

    logging.info(f"Data - Processed data")

    # Split the dataset to test for the generalisation form
    dataset = uniform_split_dataset(
        dataset=dataset,
        seed=train_config["split"]["seed"],
        num_folds=train_config["split"]["num_folds"],
        val_to_test_ratio=train_config["split"]["val_to_test_ratio"],
        cur_fold=args.fold,
    )[args.split.split_name]

    # For some reason the sorting now hangs the script
    # Not needed for testing, article_id is output as well
    #dataset = dataset.sort(column_names=["article_id"])
    #logging.info(f"Data - Sorted dataset on article_id column")

    logging.info(f"Data - Split dataset into separate splits")

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

    logging.info("Model - Finished loading model")

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
    model_outputs_file_name = (
        f"model_outputs_year[{args.year}]_split[{args.split.split_name}].csv"
    )

    (checkpoints_dir / model_outputs_file_name).unlink(missing_ok=True)

    model, dataloader = accelerator.prepare(model, dataloader)

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
            logging.info(
                f"Evaluation - {i} / {len(dataloader)} [{(i / len(dataloader)) * 100:.2f}%]"
            )

    logging.info("Evaluation - Uploading artifacts")
    task.upload_artifact(
        name=model_outputs_file_name,
        artifact_object=checkpoints_dir / model_outputs_file_name,
    )

    logging.info("Finished evaluation")


if __name__ == "__main__":
    test()
