from collections import defaultdict
from pathlib import Path
import os
import math
import typing
import re
import json
import logging
import sys
import time
import datetime

import dotenv
import numpy as np
import duckdb
import datasets
import mosestokenizer
import pydantic
import omegaconf
import hydra
import openai
from openai.lib._parsing._completions import type_to_response_format_param

from misinfo_general.utils import BaseConfig, log_config

dotenv.load_dotenv()

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant, helping analyse the properties of news articles. Before a final answer, make sure to explain your thinking."

USER_PROMPT_PREFIX = """
Please classify how objective the following article is.
Objective articles take a neutral stance on topics, and focus on reporting factual news.
Subjective articles instead focus on opinions, which are more difficult to verify and can take specific stances for or against topics.
The title and body are provided. After you provide your reasoning, respond with one of {entirely objective, mostly objective, mixed, mostly subjective, entirely subjective}, and nothing else.
"""


class ObjectivityAnnotationConfig(BaseConfig):
    class Task(pydantic.BaseModel):
        endpoint: str
        completion_window: str
        model: str
        model_params: typing.Optional[typing.Dict[str, typing.Any]]

    seed: int
    year: pydantic.conint(ge=2017, le=2022)  # type: ignore

    batch_size: pydantic.PositiveInt

    waiting_time: pydantic.PositiveInt

    dataset_loc: pydantic.DirectoryPath
    metadata_db_loc: pydantic.FilePath

    included_articles_start: pydantic.NonNegativeInt
    included_articles_end: pydantic.NonNegativeInt
    context_length: pydantic.PositiveInt

    output_loc: pydantic.NewPath | pydantic.DirectoryPath

    task: typing.Optional[Task]


class ReasoningOutput(pydantic.BaseModel):
    explanation: str
    final_answer: str


def generate_subset_dataset(
    seed: int,
    year: int,
    included_articles_start: int,
    included_articles_end: int,
    dataset_loc: os.PathLike,
    metadata_db_loc: os.PathLike,
) -> datasets.Dataset:
    logger = logging.getLogger()

    # Generate a RNG for reproducibility
    rng = np.random.default_rng(seed=seed)

    logger.info(f"Set RNG to seed {seed}. State: {rng.bit_generator.state}")

    # Convert the locations to paths
    dataset_loc = Path(dataset_loc)
    metadata_db_loc = Path(metadata_db_loc)

    # Access the metadata database
    metadata_db_fp = str(metadata_db_loc.resolve())
    metadata_db = duckdb.connect(database=metadata_db_fp, read_only=True)

    logger.info(f"Connected to metadata database at: {metadata_db_fp}.")

    # ==============================================================================
    # Fetch the article IDs and shuffle their order
    # ==============================================================================
    source_article_ids = metadata_db.sql(
        f"""
        SELECT articles_year.source, article_id
        FROM (
            SELECT *
            FROM articles
            WHERE year = {year}
        ) AS articles_year
            INNER JOIN sources
            ON articles_year.source = sources.source
        ORDER BY articles_year.source, article_id
        """
    ).fetchall()

    source_to_article_ids = defaultdict(list)

    for source, article_id in source_article_ids:
        source_to_article_ids[source].append(article_id)

    source_to_article_ids = dict(source_to_article_ids)

    logger.info(
        f"Generated publisher to article mapping. Found {len(source_to_article_ids)} publishers."
    )

    # ==============================================================================
    # Keep only q subset of the article IDs
    # ==============================================================================
    subset_article_ids = []
    for source, article_ids in source_to_article_ids.items():
        # First shuffle the articles
        shuffled_article_ids = rng.permuted(article_ids).tolist()

        # Then add the first N elements to a list
        subset_article_ids.extend(
            shuffled_article_ids[included_articles_start:included_articles_end]
        )

    subset_article_ids = set(sorted(subset_article_ids))

    logger.info("Subsampled article ids.")

    # ==============================================================================
    # Load in and filter then dataset
    # ==============================================================================
    logger.info("Loading dataset from disk.")

    dataset = datasets.Dataset.load_from_disk(str(dataset_loc.resolve() / str(year)))

    logger.info("Filter dataset to only subsampled article ids.")

    subset_dataset = dataset.filter(lambda x: x["article_id"] in subset_article_ids)

    logger.info("Finished filtering dataset.")

    return subset_dataset


def construct_batch_row(
    config: ObjectivityAnnotationConfig,
    article: typing.Dict[str, typing.Any],
    context_length: int,
    special_token_finder,
    tokenizer,
) -> typing.Dict:
    def construct_prompt(
        article: typing.Dict[str, typing.Any], context_length: int
    ) -> str:
        article_content = special_token_finder.sub("… ", article["content"])

        article_content_tokens = tokenizer.tokenize(article_content)[:-1] + [" …"]

        if len(article_content_tokens) <= context_length:
            article_content_detokens = article_content
        else:
            article_content_detokens = (
                tokenizer.detokenize(
                    article_content_tokens[: math.floor(context_length / 2)]
                )
                + " … "
                + tokenizer.detokenize(
                    article_content_tokens[-math.ceil(context_length / 2) :]
                )
            )

        prompt_data = f"Title: {article['title']}\nBody: {article_content_detokens}"

        full_prompt = USER_PROMPT_PREFIX.strip()
        full_prompt += f"\n\n{prompt_data}:"

        return full_prompt

    batch_row = {
        "custom_id": article["article_id"],
        "method": "POST",
        "url": config.task.endpoint,
        "body": {
            "model": config.task.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": construct_prompt(
                        article=article, context_length=context_length
                    ),
                },
            ],
            "response_format": type_to_response_format_param(ReasoningOutput),
            **config.task.model_params,
        },
    }

    return batch_row


@hydra.main(
    version_base="1.3", config_path="../config", config_name="objectivity_annotation"
)
def annotate_articles(parsed_config: omegaconf.DictConfig) -> None:
    # ==========================================================================
    # Setup
    # ==========================================================================
    config: ObjectivityAnnotationConfig = ObjectivityAnnotationConfig.from_hydra(
        parsed_config
    )

    logging.basicConfig(
        format="[%(asctime)s - %(relativeCreated)9d] %(levelname)-8s | %(module)s:%(lineno)s:%(funcName)s | %(message)s",
        level=logging.DEBUG if config.debug else logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
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
    run_timestamp = datetime.datetime.fromtimestamp(time.time()).strftime(
        "%Y%m%d%H%M%S"
    )

    logger.info(f"Timestamp: {run_timestamp}")

    logger.info("Generating dataset sample.")

    subset_dataset = generate_subset_dataset(
        seed=config.seed,
        year=config.year,
        included_articles_start=config.included_articles_start,
        included_articles_end=config.included_articles_end,
        dataset_loc=config.dataset_loc,
        metadata_db_loc=config.metadata_db_loc,
    )

    logger.info(msg="Finished generating dataset sample.")
    logger.info(msg=f"Found {subset_dataset.num_rows} articles.")

    # ======================================================================
    # Generate the prompts
    # ======================================================================
    logger.info(msg="Generating prompts.")

    special_token_finder = re.compile(
        r"(((<copyright>)|(<selfref>)|(<selfref>)|(<url>)) *)+"
    )

    tokenizer = mosestokenizer.MosesTokenizer(lang="en")

    estimated_total_num_tokens = 0
    batch = []
    for row in subset_dataset:
        formatted_row = construct_batch_row(
            config=config,
            article=row,
            context_length=config.context_length,
            special_token_finder=special_token_finder,
            tokenizer=tokenizer,
        )

        batch.append(formatted_row)

        estimated_total_num_tokens += 0.75 * len(
            formatted_row["body"]["messages"][0]["content"]
        )
        estimated_total_num_tokens += 0.75 * len(
            formatted_row["body"]["messages"][1]["content"]
        )

    logger.info(msg="Finished generating prompts.")
    logger.info(
        msg=f"An estimated total of {estimated_total_num_tokens:.2e} input tokens."
    )

    if config.dry_run:
        raise KeyboardInterrupt()

    num_batches = math.ceil(len(batch) / config.batch_size)

    client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    for batch_num in range(num_batches):
        logger.info(f"Processing mini batch {batch_num}/{num_batches}.")

        mini_batch = batch[
            batch_num * config.batch_size : (batch_num + 1) * config.batch_size
        ]

        # ==================================================================
        # Dump the data
        # ==================================================================
        logger.info(msg=f"{batch_num}/{num_batches} - Dumping data.")

        os.makedirs(config.output_loc / f"{run_timestamp}", exist_ok=True)

        input_fp = (
            config.output_loc / f"{run_timestamp}" / f"input_{batch_num}.jsonl"
        ).resolve()
        with open(input_fp, "w") as f:
            for batch_row in mini_batch:
                jout = json.dumps(batch_row) + "\n"
                f.write(jout)

        logger.info(msg=f"{batch_num}/{num_batches} - Finished dumping data.")
        logger.info(
            msg=f"{batch_num}/{num_batches} - File can be found at: {str(input_fp)}."
        )

        # ==================================================================
        # Uploading mini batch to OpenAI
        # ==================================================================
        logger.info(msg=f"{batch_num}/{num_batches} - Uploading batch file to OpenAI.")

        batch_input_file = client.files.create(
            file=open(str(input_fp), "rb"), purpose="batch"
        )

        # ======================================================================
        # Submit batch job
        # ======================================================================
        logger.info(msg=f"{batch_num}/{num_batches} - Submitting batch job.")

        submitted_mini_batch = client.batches.create(
            input_file_id=batch_input_file.id,
            endpoint=config.task.endpoint,
            completion_window=config.task.completion_window,
        )

        mini_batch_id = submitted_mini_batch.id

        logger.info(
            msg=f"{batch_num}/{num_batches} - Submitted batch job: {mini_batch_id}."
        )

        while True:
            time.sleep(config.waiting_time)
            logger.info(
                msg=f"{batch_num}/{num_batches} - Checking status of mini batch {mini_batch_id}"
            )

            submitted_mini_batch = client.batches.retrieve(mini_batch_id)

            status = submitted_mini_batch.status

            logger.info(f"{batch_num}/{num_batches} - Current status: {status}")

            if status == "failed":
                raise KeyboardInterrupt("Job failed...")
            elif status == "completed":
                logger.info(
                    f"{batch_num}/{num_batches} - Finished processing mini batch."
                )

                output_file_id = submitted_mini_batch.output_file_id

                output_file = client.files.content(file_id=output_file_id)

                output_fp = (
                    config.output_loc / f"{run_timestamp}" / f"output_{batch_num}.jsonl"
                ).resolve()
                with open(output_fp, "wb") as f:
                    f.write(output_file.content)

                logger.info(f"{batch_num}/{num_batches} - Fetched output file.")

                break

    return None


if __name__ == "__main__":
    annotate_articles()
