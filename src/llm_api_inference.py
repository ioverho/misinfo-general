import asyncio
import dataclasses
import datetime
import json
import logging
import math
import os
import pathlib
import re
import sys
import tempfile
import time
import typing
from collections import defaultdict

import dotenv
import numpy as np
import duckdb
import datasets
import openai
import tqdm
import transformers
import mosestokenizer
import pydantic
import omegaconf
import hydra
import google
from google import genai

from misinfo_general.utils import BaseConfig, log_config

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

HF_ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]

SYSTEM_PROMPT = """You are a helpful assistant, moderating news articles. Your task is to identify articles from unreliable publishers.

Please analyse the properties of the following articles.

Label it as reliable if you think it's likely it came from a reliable news publisher.
Reliable news publishers are trustworthy, and report on news without an overly biased framing or bias, and will not include misinformation.

Label it as unreliable if you think it's likely it came from an unreliable news publisher.
Unreliable news publishers are not trustworthy, are strongle biased, and might discuss misinformation.

Your final answer should be either 'reliable' or 'unreliable'.
""".strip()


@dataclasses.dataclass
class ModelConfig:
    provider: str
    model: str
    max_tokens: int
    thinking_budget: int
    include_thoughts: bool


class LLMAPIConfig(BaseConfig):
    dataset_loc: pydantic.DirectoryPath
    metadata_db_loc: pydantic.FilePath
    output_loc: str

    seed: int

    min_publisher_topic_occurence: int = 1
    max_publisher_topic_occurence: int = 1

    k_shot: int = 0
    context_length: int = 512

    num_proc: int | None = None

    model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig)

    batch_size: int = 128
    batch_initial_index: int = 0
    batch_terminal_index: int | None = None
    dataset_limit: int | None = None


SPECIAL_TOKEN_FINDER = re.compile(
    r"(((<copyright>)|(<selfref>)|(<selfref>)|(<url>)|(<twitter>)) *)+"
)

TOKENIZER = mosestokenizer.MosesTokenizer(lang="en")

HF_TOKENIZER = transformers.AutoTokenizer.from_pretrained(
    pretrained_model_name_or_path="meta-llama/Meta-Llama-3-8B-Instruct",
    padding_side="left",
    token=HF_ACCESS_TOKEN,
)


def filter_on_article_ids(row, article_ids: set[str]) -> bool:
    return row["article_id"] in article_ids


def process_article_content(
    article: dict[str, typing.Any], context_length: int
) -> dict[str, typing.Any]:
    article_content = SPECIAL_TOKEN_FINDER.sub("… ", article["content"])

    article_content_tokens = TOKENIZER.tokenize(article_content)[:-1] + [" …"]

    if len(article_content_tokens) <= context_length:
        article_content_detokens = article_content
    else:
        article_content_detokens = TOKENIZER.detokenize(
            article_content_tokens[:context_length]
        )

    return {"content": article_content_detokens}


def exemplar_to_str(exemplar: tuple[dict[str, typing.Any], str]):
    exemplar_article, label = exemplar

    return f"Article: {exemplar_article['content']}\n\nLabel: {label}"


def fetch_article_from_article_id(
    article_id: str,
    dataset_dict: datasets.DatasetDict,
    id_mapping: dict[int, dict[str, int]],
) -> dict[str, typing.Any]:
    year, _ = article_id.split("-")

    dataset_id = id_mapping[year][article_id]

    return dataset_dict[year][dataset_id]


@dataclasses.dataclass
class Prompt:
    article_id: str
    content: str


async def deepseek_process_single_prompt(article_id: str, client, **kwargs) -> str:
    try:
        response = await client.chat.completions.create(**kwargs)

        response_dict = response.to_dict()
    except Exception as e:
        response_dict = {"exception": str(e)}

    response_dict["article_id"] = article_id

    response_json = json.dumps(response_dict)

    return response_json


async def deepseek_process_batch(
    client, system_prompt: str, prompts: list[Prompt], **kwargs
) -> list:
    results = []

    tasks = [
        deepseek_process_single_prompt(
            article_id=prompt.article_id,
            client=client,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt.content},
            ],
            response_format={"type": "json_object"},
            **kwargs,
        )
        for prompt in prompts
    ]

    # Gather and run the tasks concurrently
    results = await asyncio.gather(*tasks)

    return results


def gemini_construct_batch_row(
    prompt: Prompt,
    model: str,
    thinking_budget: int = 1024,
    include_thoughts: bool = False,
    **kwargs,
) -> dict:
    batch_row = {
        "key": prompt.article_id,
        "request": {
            "model": f"models/{model}",
            "contents": [{"role": "user", "parts": [{"text": prompt.content}]}],
            "systemInstruction": {"parts": [{"text": SYSTEM_PROMPT}]},
            "generationConfig": {
                "temperature": 0.0,
                "thinkingConfig": {
                    "thinkingBudget": thinking_budget,
                    "includeThoughts": include_thoughts,
                },
                #"responseMimeType": "text/x.enum",
                #"responseSchema": {
                #    "type": "STRING",
                #    "enum": ["reliable", "unreliable"],
                #},
            },
        },
    }

    return batch_row


@hydra.main(
    version_base="1.3", config_path="../config", config_name="llm_api_inference"
)
def annotate_articles(parsed_config: omegaconf.DictConfig) -> None:
    run_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%Y%m%d%H%M%S"
    )

    # ==========================================================================
    # Setup
    # ==========================================================================
    config: LLMAPIConfig = LLMAPIConfig.from_hydra(parsed_config)

    output_dir = pathlib.Path(config.output_loc)
    output_dir.mkdir(exist_ok=True, parents=True)

    log_file_path = output_dir / f"{run_timestamp}.log"

    logging.basicConfig(
        format="[%(asctime)s - %(relativeCreated)9d] %(levelname)-8s | %(module)s:%(lineno)s:%(funcName)s | %(message)s",
        level=logging.DEBUG if config.debug else logging.INFO,
        handlers=[
            logging.StreamHandler(stream=sys.stdout),
            logging.FileHandler(filename=log_file_path, mode="w", encoding="utf8"),
        ],
    )

    logger.info("Parsed config.")

    if config.print_args:
        log_config(config)

    if config.disable_progress_bar:
        datasets.disable_progress_bar()
        logger.info("Disabled `datasets` progress bar.")

    # ======================================================================
    # Load the dataset and subset using stratified publisher sampling
    # ======================================================================
    logger.info(f"Timestamp: {run_timestamp}")

    logger.info("Generating dataset sample.")

    # Load in the metadata database ============================================
    metadata_db = duckdb.connect(database=config.metadata_db_loc, read_only=True)

    # Find all reliable and unreliable publishers
    unreliable_publishers = {
        publisher
        for (publisher,) in metadata_db.sql(
            """
            SELECT source
            FROM sources
            WHERE label = 'Satire' OR label = 'Questionable Source' OR label = 'Conspiracy-Pseudoscience'
            """
        ).fetchall()
    }

    reliable_publishers = {
        publisher
        for (publisher,) in metadata_db.sql(
            """
            SELECT source
            FROM sources
            WHERE NOT (label = 'Satire' OR label = 'Questionable Source' OR label = 'Conspiracy-Pseudoscience')
            """
        ).fetchall()
    }

    # Load in the dataset splits ===============================================
    dataset_splits = datasets.DatasetDict.load_from_disk(
        dataset_dict_path=config.dataset_loc
    )

    # ======================================================================
    # Generate (topic_id, year) to publisher to articles mapping
    # ======================================================================
    # Generate a mapping for (topic_id, year) pairs to publishers and articles
    topic_year_publisher_articles = metadata_db.sql(
        query="""
        SELECT imbalanced_hyper_cluster, year, source, ARRAY_AGG(article_id ORDER BY article_id ASC)
        FROM articles NATURAL JOIN topics
        GROUP BY source, year, imbalanced_hyper_cluster
        ORDER BY source, year, imbalanced_hyper_cluster
        """
    ).fetchall()

    topic_year_to_publisher_to_article_ids = defaultdict(dict)

    for topic_cluster, year, publisher, article_ids in topic_year_publisher_articles:
        topic_year_to_publisher_to_article_ids[(year, topic_cluster)][publisher] = (
            article_ids
        )

    topic_year_to_publisher_to_article_ids = dict(
        topic_year_to_publisher_to_article_ids
    )

    # ==========================================================================
    # Subsample the datasets
    # ==========================================================================
    rng = np.random.default_rng(config.seed)

    topic_year_to_publisher_to_subsample = defaultdict(dict)

    for (
        year_topic_id,
        publisher_sub_dict,
    ) in topic_year_to_publisher_to_article_ids.items():
        for publisher, article_ids in publisher_sub_dict.items():
            n_samples = rng.integers(
                low=min(config.min_publisher_topic_occurence, len(article_ids)),
                high=min(
                    config.max_publisher_topic_occurence,
                    len(article_ids),
                )
                + 1,
            )

            samples = rng.choice(article_ids, size=n_samples, replace=False)

            topic_year_to_publisher_to_subsample[year_topic_id][publisher] = (
                samples.tolist()
            )

    topic_year_to_publisher_to_subsample = dict(topic_year_to_publisher_to_subsample)

    # Flatten the mapping and add k-shot exemplars
    records = []
    for topic_year, publisher_subdict in topic_year_to_publisher_to_subsample.items():
        for publisher, subsample in publisher_subdict.items():
            other_reliable_publishers = (
                set(publisher_subdict.keys()) & reliable_publishers
            ) - {publisher}
            other_unreliable_publishers = (
                set(publisher_subdict.keys()) & unreliable_publishers
            ) - {publisher}

            for article in subsample:
                if config.k_shot != 0:
                    chosen_reliable_sources = rng.choice(
                        list(other_reliable_publishers),
                        size=config.k_shot,
                        replace=False,
                    )
                    chosen_unreliable_sources = rng.choice(
                        list(other_unreliable_publishers),
                        size=config.k_shot,
                        replace=False,
                    )

                    reliable_exemplars = [
                        article_id
                        for source in chosen_reliable_sources
                        for article_id in rng.choice(
                            topic_year_to_publisher_to_subsample[topic_year][source],
                            size=1,
                            replace=False,
                        ).tolist()
                    ]

                    unreliable_exemplars = [
                        article_id
                        for source in chosen_unreliable_sources
                        for article_id in rng.choice(
                            topic_year_to_publisher_to_subsample[topic_year][source],
                            size=1,
                            replace=False,
                        ).tolist()
                    ]

                else:
                    reliable_exemplars = []
                    unreliable_exemplars = []

                records.append(
                    {
                        "article_id": article,
                        "reliable_exemplars": reliable_exemplars,
                        "unreliable_exemplars": unreliable_exemplars,
                    }
                )

    records = sorted(records, key=lambda x: x["article_id"])

    logger.info(msg=f"Found {len(records)} total main articles.")

    # Collect all articles
    # Includes main article and all exemplars
    all_subsample_articles = (
        {record["article_id"] for record in records}
        | {
            article_id
            for record in records
            for article_id in record["reliable_exemplars"]
        }
        | {
            article_id
            for record in records
            for article_id in record["unreliable_exemplars"]
        }
    )

    subsampled_dataset = dataset_splits.filter(
        function=lambda x: x["article_id"] in all_subsample_articles,
        batched=False,
        num_proc=config.num_proc,
    )

    logger.info(
        msg=f"Found {subsampled_dataset.num_rows} total main + exemplars articles."
    )

    logger.info(msg="Pre-processing the dataset")

    processed_dataset = subsampled_dataset.map(
        process_article_content,
        batched=False,
        num_proc=config.num_proc,
        fn_kwargs=dict(context_length=config.context_length),
    ).sort(column_names=["article_id"])

    # Generate a new article_id to dataset_id mapping ==========================
    article_id_to_processed_dataset_id = {
        str(year): {
            article_id: i
            for i, article_id in enumerate(processed_dataset[str(year)]["article_id"])
        }
        for year in range(2017, 2023)
    }

    logger.info(msg="Finished pre-processing the dataset")

    # ======================================================================
    # Generate the prompts
    # ======================================================================
    logger.info(msg="Generating prompts.")

    prompts = []
    for record in records:
        prompt_string = ""

        if config.k_shot != 0:
            reliable_examplars = [
                fetch_article_from_article_id(
                    article_id=article_id,
                    dataset_dict=processed_dataset,
                    id_mapping=article_id_to_processed_dataset_id,
                )
                for article_id in record["reliable_exemplars"]
            ]

            unreliable_examplars = [
                fetch_article_from_article_id(
                    article_id=article_id,
                    dataset_dict=processed_dataset,
                    id_mapping=article_id_to_processed_dataset_id,
                )
                for article_id in record["unreliable_exemplars"]
            ]

            exemplars = [
                (article, label)
                for i in range(config.k_shot)
                for article, label in zip(
                    [reliable_examplars[i], unreliable_examplars[i]],
                    ["reliable", "unreliable"],
                )
            ]

            prompt_string += "\n\n===\n\n**Examples**\n\n" + "\n\n".join(
                map(exemplar_to_str, exemplars)
            )

        main_article = fetch_article_from_article_id(
            article_id=record["article_id"],
            dataset_dict=processed_dataset,
            id_mapping=article_id_to_processed_dataset_id,
        )

        prompt_string += (
            "\n\n===\n\n**Please label the following article**"
            + f"\n\nArticle: {main_article['content']}\n\nLabel:"
        )

        prompt = Prompt(article_id=record["article_id"], content=prompt_string)

        prompts.append(prompt)

    logger.info(msg="Finished generating prompts.")

    if config.dataset_limit is not None:
        prompts = prompts[: config.dataset_limit]

    # ==========================================================================
    # Figure out how many tokens there are in the dataset
    # ==========================================================================

    all_num_tokens = []
    for prompt in prompts:
        num_tokens = 0.3 * len(prompt.content)
        all_num_tokens.append(num_tokens)

    logging.info(f"Total num records: {len(prompts)}")
    logging.info(f"Total num tokens: {np.sum(all_num_tokens):.2e}")
    logging.info(f"Mean num tokens: {np.mean(all_num_tokens).tolist()}")
    logging.info(
        f"Median num tokens: {np.quantile(all_num_tokens, q=[0.5]).tolist()[0]}"
    )
    logging.info(f"Min num tokens: {np.min(all_num_tokens).tolist()}")
    logging.info(msg=f"Max num tokens: {np.max(all_num_tokens).tolist()}")

    total_num_batches = math.ceil(len(prompts) / config.batch_size)

    logging.info(msg=f"Total num batches: {total_num_batches}")

    # ==========================================================================
    # INFERENCE
    # ==========================================================================
    output_file_path = output_dir / f"{run_timestamp}.jsonl"
    output_file_path.touch()
    logging.info(f"Saving responses to: {output_file_path.resolve()}")

    if config.model.provider == "deepseek":
        logging.info("Using DeepSeek API")

        system_prompt = SYSTEM_PROMPT + (
            "\n\nMake sure your answer is formatted as a JSON object."
            "\n\nExample JSON object:\n{\n\t'Label': 'YOUR_LABEL'\n}"
        )

        client = openai.AsyncOpenAI(
            api_key=os.environ["DEEPSEEK_API_KEY"],
            base_url="https://api.deepseek.com",
        )

        for batch_num in tqdm.tqdm(iterable=range(total_num_batches)):
            if batch_num < config.batch_initial_index:
                continue
            elif (config.batch_terminal_index is not None) and (
                batch_num >= config.batch_terminal_index
            ):
                raise KeyboardInterrupt(f"Reach terminal batch number: {batch_num}")

            batch = prompts[
                batch_num * config.batch_size : (batch_num + 1) * config.batch_size
            ]

            batch_responses = asyncio.run(
                deepseek_process_batch(
                    client=client,
                    system_prompt=system_prompt,
                    prompts=batch,
                    model=config.model.model,
                    max_tokens=config.model.max_tokens,
                )
            )

            for response in batch_responses:
                with output_file_path.open("a") as f:
                    f.write(response + "\n")

    elif config.model.provider == "google":
        client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

        for batch_num in tqdm.tqdm(iterable=range(total_num_batches)):
            if batch_num < config.batch_initial_index:
                continue
            elif (config.batch_terminal_index is not None) and (
                batch_num >= config.batch_terminal_index
            ):
                raise KeyboardInterrupt(f"Reach terminal batch number: {batch_num}")

            batch = prompts[
                batch_num * config.batch_size : (batch_num + 1) * config.batch_size
            ]

            requests = [
                gemini_construct_batch_row(prompt=prompt, **config.model.__dict__)
                for prompt in batch
            ]

            with pathlib.Path("./temp/gemini_batch_file.jsonl").open("wb") as fp:
                for request in requests:
                    fp.write((json.dumps(obj=request) + "\n").encode())

                fp.close()

            with tempfile.NamedTemporaryFile(delete_on_close=False) as fp:
                for request in requests:
                    fp.write((json.dumps(obj=request) + "\n").encode())

                fp.close()

                uploaded_file = client.files.upload(
                    file=fp.name,
                    config=google.genai.types.UploadFileConfig(
                        display_name=f"batch[{batch_num}]", mime_type="jsonl"
                    ),
                )

                batch_job = client.batches.create(
                    model=f"models/{config.model.model}",
                    src=uploaded_file.name,
                    config={
                        "display_name": f"batch-{batch_num}",
                    },
                )

            logging.info(f"Polling status for job: {batch_job.name}")
            batch_job = client.batches.get(name=batch_job.name)  # Initial get
            while batch_job.state.name not in {
                "JOB_STATE_SUCCEEDED",
                "JOB_STATE_FAILED",
                "JOB_STATE_CANCELLED",
            }:
                logging.info(f"Current state: {batch_job.state.name}")
                time.sleep(30)  # Wait for 30 seconds before polling again
                batch_job = client.batches.get(name=batch_job.name)

            logging.info(f"Job finished with state: {batch_job.state.name}")
            if batch_job.state.name == "JOB_STATE_FAILED":
                logging.error(f"Error: {batch_job.error}")
            else:
                # Results are in a file
                result_file_name = batch_job.dest.file_name
                logging.info(f"Results are in file: {result_file_name}")

                logging.info("Downloading result file content...")
                file_content = client.files.download(file=result_file_name)

            with output_file_path.open("a") as f:
                f.write(file_content.decode("utf-8"))

            #raise KeyboardInterrupt()


    logging.info("Script finished succesfully!")


if __name__ == "__main__":
    annotate_articles()
