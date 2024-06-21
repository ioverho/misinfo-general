import os
import logging
import math
import sys
from pathlib import Path

from clearml import Task
from omegaconf import DictConfig, OmegaConf
import datasets
import hydra
import numpy as np
import sklearn.metrics as metrics
import transformers
import wandb
import datasets
from datasets import concatenate_datasets

from misinfo_benchmark_models import SPECIAL_TOKENS
from misinfo_benchmark_models.experiment_metadata import ExperimentMetaData
from misinfo_benchmark_models.labelling import MBFCBinaryLabeller
from misinfo_benchmark_models.data import process_dataset, collator, static_collator
from misinfo_benchmark_models.splitting import (
    uniform_split_dataset,
    publisher_split_dataset,
    topic_split_dataset,
)
from misinfo_benchmark_models.metrics import compute_clf_metrics
from misinfo_benchmark_models.utils import print_config, save_config


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def train(args: DictConfig):
    # ==========================================================================
    # Setup
    # ==========================================================================
    # Setup directories for saving output and logs
    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists()
    assert (data_dir / "hf").exists()
    assert (data_dir / "db").exists() or (data_dir / "db_export").exists()

    # Check additional arg rules
    assert args.generalisation_form is not None
    assert isinstance(args.fold, int)
    assert args.fold >= 0 and args.fold < args.split.num_folds

    experiment_metadata = ExperimentMetaData.from_args(
        args=args, generalisation_form=args.generalisation_form
    )

    # ClearML logging ==========================================================
    Task.set_random_seed(args.seed)

    task = Task.init(
        project_name=experiment_metadata.project_name,
        task_name=experiment_metadata.task_name,
        task_type="training",
    )

    task.connect(OmegaConf.to_container(args, resolve=True))

    # Save checkpoints to a directory with the ClearML ID to align everything
    checkpoints_dir = (Path(args.checkpoints_dir) / f"{task.task_id}").resolve()
    os.makedirs(name=checkpoints_dir, exist_ok=True)

    # Setup logging
    logging.basicConfig(
        format="[%(asctime)s] %(levelname)s: %(message)s",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(str(checkpoints_dir / "log.txt")),
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Print config for logging purposes
    print_config(args, logger=logging)

    save_config(args, results_dir=checkpoints_dir)

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
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
    )

    tokenizer.model_max_length = args.data.max_length

    new_num_tokens = tokenizer.add_tokens(
        list(SPECIAL_TOKENS.values()),
        special_tokens=True,
    )
    logging.info("Data - Fetched tokenizer")
    logging.info(f"Data - Added {new_num_tokens} new tokens")

    if args.generalisation_form == "covid":
        
        dataset = datasets.DatasetDict({
            "train": process_dataset(
                    data_dir=data_dir,
                    year="covid_train",
                    model_name=args.model_name,
                    max_length=args.data.max_length,
                    batch_size=args.batch_size.tokenization,
                    tokenizer=tokenizer,
                    labeller=labeller,
                    logger=logging,
                ),
            
            "val": process_dataset(
                    data_dir=data_dir,
                    year="covid_val",
                    model_name=args.model_name,
                    max_length=args.data.max_length,
                    batch_size=args.batch_size.tokenization,
                    tokenizer=tokenizer,
                    labeller=labeller,
                    logger=logging,
                ),
            
            "test": process_dataset(
                    data_dir=data_dir,
                    year="covid_test",
                    model_name=args.model_name,
                    max_length=args.data.max_length,
                    batch_size=args.batch_size.tokenization,
                    tokenizer=tokenizer,
                    labeller=labeller,
                    logger=logging,
                ),
        })

        logging.info("Data - Concatenated all COVID datasets together")

    else:
        dataset = process_dataset(
            data_dir=data_dir,
            year=args.year,
            model_name=args.model_name,
            max_length=args.data.max_length,
            batch_size=args.batch_size.tokenization,
            tokenizer=tokenizer,
            labeller=labeller,
            logger=logging,
        )

        logging.info("Data - Processed dataset")

    # Split the dataset
    if args.generalisation_form == "uniform":
        dataset_splits = uniform_split_dataset(
            dataset=dataset,
            seed=args.split.seed,
            num_folds=args.split.num_folds,
            val_to_test_ratio=args.split.val_to_test_ratio,
            cur_fold=args.fold,
        )

    elif args.generalisation_form == "publisher":
        dataset_splits = publisher_split_dataset(
            dataset=dataset,
            db_loc="./data/db/misinformation_benchmark_metadata.db",
            seed=args.seed,
            year=args.year,
            val_prop=args.split.val_prop,
            test_prop=args.split.test_prop,
        )

    elif args.generalisation_form == "topic":
        dataset_splits = topic_split_dataset(
            dataset=dataset,
            db_loc="./data/db/misinformation_benchmark_metadata.db",
            seed=args.seed,
            year=args.year,
            val_prop=args.split.val_prop,
            test_prop=args.split.test_prop,
        )

    elif args.generalisation_form == "covid":
        # COVID is special, have already done the text-processing and splitting
        dataset_splits = dataset

    logging.info("Data - Finished splitting dataset")
    logging.info(f"Data - Train size: {dataset_splits['train'].num_rows}")
    logging.info(f"Data - Val   size: {dataset_splits['val'].num_rows}")
    logging.info(f"Data - Test  size: {dataset_splits['test'].num_rows}")

    dataset_splits.set_format(
        type="torch", columns=["input_ids", "attention_mask", "num_tokens", "label"]
    )

    # ==========================================================================
    # Model loading
    # ==========================================================================
    model = transformers.AutoModelForSequenceClassification.from_pretrained(
        args.model_name,
        num_labels=len(labeller.int_to_label),
        ignore_mismatched_sizes=True,
        **args.model,
    )

    # Add special tokens to the embeddings layer
    model.resize_token_embeddings(
        new_num_tokens=model.get_input_embeddings().weight.shape[0] + new_num_tokens,
        pad_to_multiple_of=64,
    )

    # Freeze all layers of the model except for the embeddings layer
    for p in model.deberta.parameters():
        p.requires_grad = False

    for p in model.deberta.embeddings.parameters():
        p.requires_grad = True

    # ==========================================================================
    # Trainer configuration
    # ==========================================================================
    num_batches = math.ceil(args.trainer.max_steps / args.batch_size.train)
    logging.info(f"Expected number of steps: {num_batches}")

    expected_num_epochs = args.trainer.max_steps / dataset_splits["train"].num_rows
    logging.info(f"Expected number of epochs: {expected_num_epochs:.2f}")

    eval_batches = max(1, int(args.trainer.eval_prop * num_batches))
    logging.info(f"Eval every: {eval_batches}")

    log_batches = max(1, int(args.trainer.logging_prop * num_batches))
    logging.info(f"Logging every: {log_batches}")

    num_warmup_steps = round(args.optim.warmup_ratio * num_batches)
    logging.info(f"Number of warmup steps: {num_warmup_steps}")

    optimizer = transformers.AdamW(
        params=[
            {
                "params": model.deberta.embeddings.parameters(),
                "lr": args.optim.lrs.embeddings,
            },
            {"params": model.pooler.parameters(), "lr": args.optim.lrs.pooler},
            {"params": model.classifier.parameters(), "lr": args.optim.lrs.classifier},
        ],
        weight_decay=args.optim.weight_decay,
        betas=(args.optim.beta1, args.optim.beta2),
        eps=args.optim.eps,
    )

    lr_scheduler = transformers.get_scheduler(
        name=args.optim.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_batches,
        scheduler_specific_kwargs=args.optim.scheduler_kwargs,
    )

    if args.trainer.wandb:
        wandb.init(
            config=experiment_metadata.config,
            group=experiment_metadata.generalisation_form,
            job_type=str(experiment_metadata.dataset_year),
            allow_val_change=False,
            **args.logging,
        )

    training_args = transformers.TrainingArguments(
        output_dir=checkpoints_dir,
        do_train=True,
        do_eval=True,
        evaluation_strategy="steps",
        eval_steps=eval_batches,
        per_device_train_batch_size=args.batch_size.train,
        per_device_eval_batch_size=args.batch_size.eval,
        max_steps=num_batches,
        log_level="info",
        logging_strategy="steps",
        logging_steps=log_batches,
        logging_first_step=True,
        save_strategy="steps",
        save_steps=eval_batches,
        save_total_limit=1,
        save_safetensors=True,
        save_only_model=True,
        seed=args.trainer.seed,
        load_best_model_at_end=True,
        report_to=(
            []
            + (["clearml"] if args.trainer.clearml else [])
            + (["wandb"] if args.trainer.wandb else [])
        ),
        skip_memory_metrics=not args.trainer.memory_metrics,
        torch_compile=args.trainer.torch_compile,
        disable_tqdm=args.disable_progress_bar,
        **(args.trainer.kwargs if args.trainer.kwargs is not None else {}),
    )

    # ==========================================================================
    # TRAIN
    # ==========================================================================
    trainer = transformers.Trainer(
        model=model,
        args=training_args,
        data_collator=static_collator if args.trainer.torch_compile else collator,
        train_dataset=dataset_splits["train"],
        eval_dataset=dataset_splits["val"],
        compute_metrics=compute_clf_metrics,
        optimizers=(optimizer, lr_scheduler),
        callbacks=[
            transformers.EarlyStoppingCallback(
                **args.optim.early_stopping_kwargs,
            )
        ],
    )

    trainer.train()

    predict_output = trainer.predict(test_dataset=dataset_splits["test"])

    conf_mat = metrics.confusion_matrix(
        y_true=np.argmax(predict_output.predictions, axis=1),
        y_pred=predict_output.label_ids,
    )

    np.savetxt(
        fname=checkpoints_dir / "test_split_conf_mat.csv",
        X=conf_mat.astype(int),
        delimiter=",",
        encoding="utf8",
        fmt="%d",
    )

    task.upload_artifact(
        name="test_split_conf_mat",
        artifact_object=checkpoints_dir / "test_split_conf_mat.csv",
    )


if __name__ == "__main__":
    train()
