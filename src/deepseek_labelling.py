import asyncio
import dataclasses
import math
import os
import logging
import pickle
import pathlib
import re
import sys

import dotenv
import hydra
from matplotlib.pyplot import annotate
import mosestokenizer
import omegaconf
import openai
import pydantic

from misinfo_general.utils import BaseConfig

logger = logging.getLogger(__name__)

dotenv.load_dotenv()

HF_ACCESS_TOKEN = os.environ["HF_ACCESS_TOKEN"]

SYSTEM_PROMPT = "You are a helpful assistant, moderating news articles. Your task is to identify articles from unreliable publishers."

USER_PROMPT_PREFIX = """
Please analyse the properties of the following articles.

Label it as reliable if you think it's likely it came from a reliable news publisher.
Reliable news publishers are trustworthy, and report on news without an overly biased framing or bias, and will not include misinformation.

Label it as unreliable if you think it's likely it came from an unreliable news publisher.
Unreliable news publishers are not trustworthy, are strongle biased, and might discuss misinformation.

Your final answer should be either 'reliable' or 'unreliable'.
""".strip()


class ModelConfig:
    provider: str
    model: str
    max_tokens: int


class LLMAPIConfig(BaseConfig):
    dataset_loc: pydantic.DirectoryPath
    metadata_db_loc: pydantic.FilePath
    output_loc: pydantic.FilePath

    seed: int

    min_publisher_topic_occurence: int = 1
    max_publisher_topic_occurence: int = 1

    k_shot: int = 0
    context_length: int = 512

    model: ModelConfig = dataclasses.field(default_factory=lambda: ModelConfig)

    batch_size: int = 128
    dataset_limit: int | None = None


SPECIAL_TOKEN_FINDER = re.compile(
    r"(((<copyright>)|(<selfref>)|(<selfref>)|(<url>)|(<twitter>)) *)+"
)

TOKENIZER = mosestokenizer.MosesTokenizer(lang="en")


def split_prompt(prompt: str) -> None | tuple[str, str]:
    output = prompt.split("===")

    if len(output) <= 1:
        return None
    elif len(output) > 2:
        system_message = output[0]
        article = "===".join(output[1:])
    else:
        system_message, article = output

        return system_message, article


async def process_single_prompt(client, **kwargs):
    response = await client.chat.completions.create(**kwargs)

    return response


async def process_batch(
    client, system_prompt: str, prompts: list[str], **kwargs
) -> list:
    results = []

    tasks = [
        process_single_prompt(
            client=client,
            messages=[
                {
                    "role": "system",
                    "content": system_prompt,
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            **kwargs,
        )
        for prompt in prompts
    ]

    # Gather and run the tasks concurrently
    results = await asyncio.gather(*tasks)

    return results


@hydra.main(
    version_base="1.3", config_path="../config", config_name="llm_api_inference"
)
def annotate(parsed_config: omegaconf.DictConfig) -> None:
    # ==========================================================================
    # Setup
    # ==========================================================================
    config: LLMAPIConfig = LLMAPIConfig.from_hydra(parsed_config)

    logging.basicConfig(
        format="[%(asctime)s - %(relativeCreated)9d] %(levelname)-8s | %(module)s:%(lineno)s:%(funcName)s | %(message)s",
        level=logging.DEBUG if config.debug else logging.INFO,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    logger.info("Parsed config.")

    # ==========================================================================
    #
    # ==========================================================================
    fp = pathlib.Path("./temp/all_prompts.pkl")

    with fp.open("rb") as f:
        all_prompts = pickle.load(file=f)

    system_prompt = split_prompt(all_prompts[0])[0].strip()

    system_prompt += (
        "\n\nMake sure your answer is formatted as a JSON object."
        "\n\nExample JSON object:\n{\n\t'Label': 'YOUR_LABEL'\n}"
    )

    prompt_bodies = [split_prompt(prompt) for prompt in all_prompts]
    prompt_bodies = [
        prompt[1].strip() for prompt in prompt_bodies if prompt is not None
    ]

    if config.dataset_limit is not None:
        prompt_bodies = prompt_bodies[: config.dataset_limit]

    # ==========================================================================
    #
    # ==========================================================================
    client = openai.AsyncOpenAI(
        api_key=os.environ["DEEPSEEK_API_KEY"],
        base_url="https://api.deepseek.com",
    )

    total_num_batches = math.ceil(len(prompt_bodies) / config.batch_size)

    all_responses = []
    for batch_num in range(total_num_batches):
        batch = prompt_bodies[
            batch_num * config.batch_size : (batch_num + 1) * config.batch_size
        ]

        responses = asyncio.run(
            process_batch(
                client=client,
                system_prompt=system_prompt,
                prompts=batch,
                model=config.model,
                max_tokens=config.max_tokens,
            )
        )

        # TODO: process responses
        all_responses.extend(responses)

    for response in all_responses:
        print(response)


if __name__ == "__main__":
    annotate()
