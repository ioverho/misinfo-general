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
import transformers
import yaml

from misinfo_general import SPECIAL_TOKENS
from misinfo_general.experiment_metadata import ExperimentMetaData
from misinfo_general.labelling import MBFCBinaryLabeller
from misinfo_general.data import process_dataset, eval_collator


@hydra.main(version_base="1.3", config_path="../config", config_name="test_covid")
def test(args: DictConfig):
    assert args.year is not None
    assert args.checkpoint.model_name is not None
    assert args.split.split_name in ["val", "test"]

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
        task_name=f"year[all]_split[{args.split.split_name}]",
        task_type="testing",
        tags=[
            "train_year[all]",
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
        year=f"covid_{args.split.split_name}",
        model_name=train_config["model_name"],
        max_length=train_config["data"]["max_length"],
        batch_size=args.batch_size.tokenization,
        tokenizer=tokenizer,
        labeller=labeller,
        logger=logging,
    )

    dataset = dataset.sort(column_names=["article_id"])

    dataset.set_format(
        type="torch", columns=["input_ids", "attention_mask", "num_tokens", "label"]
    )

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

    # Freeze the entire model
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
    preds_file_name = f"preds_generalisation[covid]_year[{args.year}]_split[{args.split.split_name}].csv"

    (checkpoints_dir / preds_file_name).unlink(missing_ok=True)

    model, dataloader = accelerator.prepare(model, dataloader)

    for i, batch in enumerate(dataloader):
        with torch.inference_mode(True):
            logits = model.forward(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            ).logits

            preds = torch.argmax(input=logits, dim=-1).cpu()

        with open(checkpoints_dir / preds_file_name, "a") as f:
            writer = csv.writer(f)
            for row in zip(batch["article_ids"], preds.tolist(), batch["labels"]):
                writer.writerow(row)

        if i == 0 or i % 10 == 0 or i == len(dataloader) - 1:
            logging.info(
                f"Evaluation - {i} / {len(dataloader) - 1} [{(i / (len(dataloader) - 1))*100:.2f}%]"
            )

    logging.info("Evaluation - Uploading artifacts")
    task.upload_artifact(
        name=preds_file_name,
        artifact_object=checkpoints_dir / preds_file_name,
    )

    logging.info("Finished evaluation")


if __name__ == "__main__":
    test()
