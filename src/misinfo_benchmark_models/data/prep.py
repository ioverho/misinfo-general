from pathlib import Path
from functools import partial

import datasets


def tokenize_texts(content: dict, tokenizer):
    tokens = tokenizer(
        text=content,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
        return_length=True,
    )

    num_tokens = tokens["attention_mask"].sum(axis=1)

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "num_tokens": num_tokens,
    }


def map_source_to_Label(source: str, labeller):
    label = labeller(source, return_int=True)

    return {"label": label}


def process_dataset(
    data_dir,
    year: int,
    model_name: str,
    max_length: int,
    batch_size,
    tokenizer,
    labeller,
    logger,
):
    if isinstance(data_dir, str):
        data_dir = Path(data_dir)

    processed_data_name = f"year[{year}]_model[{model_name.replace('-', '_').replace('/', '-')}]_length[{max_length}]"

    processed_data_loc = data_dir / "processed" / processed_data_name

    if processed_data_loc.exists():
        try:
            dataset = datasets.Dataset.load_from_disk(str(processed_data_loc))

            logger.info(
                f"Data - Loaded existing processed dataset at: {processed_data_loc}"
            )
            logger.info("Data - Finished data processing")

            return dataset

        except Exception as e:
            logger.warning(f"Could not load existing processed dataset because: {e}")

    # Load in the data
    dataset = datasets.DatasetDict.load_from_disk(
        dataset_dict_path=str(data_dir / "hf")
    )

    dataset = dataset[str(year)]
    logger.info("Data - Fetched dataset")
    logger.info(f"Data - Data has {dataset.num_rows}")

    # Tokenize the texts
    logger.info("Data - Starting tokenization")
    dataset = dataset.map(
        partial(tokenize_texts, tokenizer=tokenizer),
        input_columns="content",
        batched=True,
        batch_size=batch_size,
    )
    logger.info("Data - Finished tokenization")

    # Map the labels to integers
    logger.info("Data - Starting label mapping")
    dataset = dataset.map(
        partial(map_source_to_Label, labeller=labeller),
        input_columns="source",
        batched=False,
    )

    feats = dataset.features.copy()
    feats["label"] = datasets.ClassLabel(
        num_classes=len(labeller.labels), names=labeller.labels
    )

    dataset = dataset.cast(feats)
    logger.info("Data - Finished label mapping")

    dataset = dataset.remove_columns(
        column_names=list(
            set(dataset.column_names)
            - {
                "article_id",
                "input_ids",
                "attention_mask",
                "num_tokens",
                "label",
            }
        )
    )

    logger.info("Data - Saving processed dataset to disk")

    dataset.save_to_disk(str(processed_data_loc))

    logger.info(f"Data - Cleaned up cache files: {dataset.cleanup_cache_files()}")

    logger.info("Data - Finished data processing")

    return dataset
