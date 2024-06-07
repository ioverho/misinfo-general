import os
import math
from pathlib import Path
import logging
import sys
import csv

import dotenv
import datasets
import transformers
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
import hydra
from clearml import Task

from misinfo_benchmark_models.utils import print_config, save_config

dotenv.load_dotenv()

os.environ["TOKENIZERS_PARALLELISM"] = "false"
HF_ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]

SYSTEM_MESSAGE = "You are a content moderator working with journalistic articles. Your task is to identify articles from unreliable publishers."
TASK_DESCRIPTION = "Does the following text come from a reliable news publisher? Respond with 'yes' or 'no'."
DOMAIN_PREMISE = "Does this article come from a reliable news publisher? 'yes' or 'no':"
ANSWERS = ["yes", "no"]


def generate_user_query(article):
    user_query = "\n\n".join(
        [
            TASK_DESCRIPTION,
            "Article: " + article,
            DOMAIN_PREMISE,
        ]
    )

    return user_query


def generate_domain_premise():
    user_query = "\n\n".join(
        [
            TASK_DESCRIPTION,
            DOMAIN_PREMISE,
        ]
    )

    return user_query


def generate_query_chat_template(article, tokenizer, **kwargs):
    full_query = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": generate_user_query(article),
            },
        ],
        **kwargs,
    )

    return full_query


def generate_domain_premise_chat(tokenizer, **kwargs):
    domain_premise_chat = tokenizer.apply_chat_template(
        [
            {
                "role": "system",
                "content": SYSTEM_MESSAGE,
            },
            {
                "role": "user",
                "content": generate_domain_premise(),
            },
        ],
        **kwargs,
    )

    return domain_premise_chat


@hydra.main(version_base="1.3", config_path="../config", config_name="llm_inference")
def inference(args: DictConfig):
    # ==========================================================================
    # Setup
    # ==========================================================================
    # Setup directories for saving output and logs
    data_dir = Path(args.data_dir).resolve()
    assert data_dir.exists()
    assert (data_dir / "hf").exists()
    assert (data_dir / "db").exists() or (data_dir / "db_export").exists()

    # ClearML logging ==========================================================
    Task.set_random_seed(args.seed)

    safe_model_name = args.model_name.replace("-", "_").replace("/", "-")

    task = Task.init(
        project_name="misinfo_benchmark_models/llama3",
        task_name=f"year[{args.year}]_model[{safe_model_name}]",
        task_type="inference",
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
            # logging.FileHandler(str(checkpoints_dir / "log.txt")),
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
    # Data loading
    # ==========================================================================
    logging.info("Data - Loading dataset")
    dataset_loc = str(Path(args.data_dir) / f"hf/{args.year}")
    dataset = datasets.Dataset.load_from_disk(dataset_loc)

    logging.info(f"Data - Found dataset at: {dataset_loc}")

    # ==========================================================================
    # Model loading
    # ==========================================================================
    # Accelerator management ===================================================
    model_loading_kwargs = dict()

    model_loading_kwargs["device_map"] = args.acceleration.device_map

    model_loading_kwargs["low_cpu_mem_usage"] = args.acceleration.low_cpu_mem_usage

    if args.acceleration.max_gpu_memory is not None:
        model_loading_kwargs["max_memory"] = {
            device_id: args.acceleration.max_gpu_memory
            for device_id in range(torch.cuda.device_count())
        }

    if args.acceleration.max_cpu_memory is not None:
        model_loading_kwargs["max_memory"] |= {
            "cpu": args.acceleration.max_cpu_memory,
        }

    # Check if we want to do quantized inference
    # Should come at no cost in inference performance, but major mem reduction
    if args.acceleration.bits_and_bytes:
        print("Using `load_in_8bit=True` to use quanitized model")
        model_loading_kwargs["load_in_8bit"] = True

    else:
        logging.info("Model - Not using quantized inference.")

    if args.acceleration.torch_dtype is not None:
        model_loading_kwargs["torch_dtype"] = args.acceleration.torch_dtype

    # Decide where we are going to be keeping the data before letting
    # the accelerator handle the forward pass
    data_device = args.acceleration.data_device

    # Model loading ============================================================
    logging.info("Model - Loading model")

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="left",
        token=HF_ACCESS_TOKEN,
    )

    article_truncator = transformers.AutoTokenizer.from_pretrained(
        args.model_name,
        padding_side="right",
        truncation_side="right",
        token=HF_ACCESS_TOKEN,
    )

    model = transformers.AutoModelForCausalLM.from_pretrained(
        args.model_name, token=HF_ACCESS_TOKEN, **model_loading_kwargs
    )

    model.eval()

    if args.model.set_pad_token:
        tokenizer.pad_token = tokenizer.eos_token
        article_truncator.pad_token = article_truncator.eos_token

    if args.model.compile:
        logging.info("Model - Compiling model")

        model = torch.compile(model, mode="default", dynamic=True)

        logging.info("Model - Finished compiling model")

    # ==========================================================================
    # Compute domain conditional
    # ==========================================================================
    logging.info("Domain Conditional - Compute domain conditional probability")

    # Run once per task
    tokenized_answers = tokenizer(ANSWERS, return_tensors="pt").input_ids[
        :, args.data.num_offset_tokens
    ]

    tokenized_domain_conditionals = tokenizer(
        generate_domain_premise_chat(
            tokenize=False,
            add_generation_prompt=True,
            tokenizer=tokenizer,
        ),
        return_tensors="pt",
        pad_to_multiple_of=64,
    )

    logging.info("Domain Conditional - Starting forward pass")

    with torch.inference_mode(True):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True, enable_math=False, enable_mem_efficient=False
        ):
            tokenized_domain_conditionals = {
                k: (obj.to(data_device) if isinstance(obj, torch.Tensor) else obj)
                for k, obj in tokenized_domain_conditionals.items()
            }

        model_output = model(**tokenized_domain_conditionals)

    domain_conditional_probabilities = F.softmax(model_output.logits, dim=1)[
        :, -1, tokenized_answers
    ].to("cpu")

    logging.info("Domain Conditional - Finished")

    # ==========================================================================
    # INFERENCE
    # ==========================================================================
    logging.info("Inference - Starting")

    num_batches = math.ceil(dataset.num_rows / args.batch_size)

    all_article_ids = []
    all_probability_decisions = []
    all_pmi_dc_decisions = []
    for batch_num in range(num_batches):
        left_edge = batch_num * args.batch_size
        right_edge = (batch_num + 1) * args.batch_size

        batch = dataset[left_edge:right_edge]
        # actual_batch_size = len(batch["content"])

        # Truncate the articles to fit within context window of LLM
        # Needs to run with right-sides padding
        tokenized_batch = article_truncator(
            batch["content"],
            return_tensors="pt",
            return_special_tokens_mask=True,
            padding="longest",
            truncation=True,
        )

        truncated_article_texts = tokenizer.batch_decode(
            tokenized_batch["input_ids"][
                :,
                args.data.num_offset_tokens : args.data.max_length
                + args.data.num_offset_tokens,
            ],
            skip_special_tokens=True,
        )

        # Convert to chat prommpts
        truncated_article_texts = [
            generate_query_chat_template(
                article=article,
                tokenizer=tokenizer,
                tokenize=False,
                add_generation_prompt=True,
            )
            for article in truncated_article_texts
        ]

        # Forward pass =========================================================
        tokenized_prompts = tokenizer(
            truncated_article_texts,
            return_tensors="pt",
            padding=True,
            pad_to_multiple_of=64,
        )

        with torch.inference_mode(True):
            with torch.backends.cuda.sdp_kernel(
                enable_flash=True, enable_math=False, enable_mem_efficient=False
            ):
                tokenized_prompts = {
                    k: (obj.to(data_device) if isinstance(obj, torch.Tensor) else obj)
                    for k, obj in tokenized_prompts.items()
                }

            model_output = model(**tokenized_prompts)

        answer_probabilities = F.softmax(model_output.logits, dim=1)[
            :, -1, tokenized_answers
        ].to("cpu")

        probability_decisions = torch.argmax(answer_probabilities, dim=1)

        pmi_dc_decisions = torch.argmax(
            answer_probabilities
            / domain_conditional_probabilities.expand(
                size=answer_probabilities.size(),
            ),
            dim=1,
        )

        all_article_ids.extend(batch["article_id"])
        all_probability_decisions.append(probability_decisions)
        all_pmi_dc_decisions.append(pmi_dc_decisions)

        if batch_num == 0 or batch_num == num_batches - 1 or (batch_num + 1) % 10 == 0:
            logging.info(
                f"Inference - Batch {batch_num}/{num_batches} [{batch_num/num_batches*100:.2f}%]"
            )

        if batch_num == 100:
            break

    logging.info("Inference - Finished")

    all_probability_decisions = torch.concat(all_probability_decisions, dim=0).tolist()
    all_pmi_dc_decisions = torch.concat(all_pmi_dc_decisions, dim=0).tolist()

    output_file_name = f"{args.year}_preds.csv"
    with open(checkpoints_dir / output_file_name, "w") as f:
        writer = csv.writer(f)

        for row in zip(
            all_article_ids, all_probability_decisions, all_pmi_dc_decisions
        ):
            writer.writerow(row)

    logging.info("Wrap up - Uploading artifacts")
    task.upload_artifact(
        name=output_file_name,
        artifact_object=checkpoints_dir / output_file_name,
    )

    logging.info("Finished.")


if __name__ == "__main__":
    inference()
