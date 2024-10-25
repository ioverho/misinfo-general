---
annotations_creators:
- expert-generated
language:
- en
language_creators:
- other
license:
- cc-by-nc-sa-4.0
multilinguality:
- monolingual
pretty_name: misinfo-general
size_categories:
- 1M<n<10M
tags:
- text
- news
- misinformation
- reliability
- '2017'
- '2018'
- '2019'
- '2020'
- '2021'
- '2022'
- datasets
- duckdb
task_categories:
- text-classification
dataset_info:
  features:
  - name: author
    dtype: string
  - name: title
    dtype: string
  - name: content
    dtype: string
  - name: source
    dtype: string
  - name: domain
    dtype: string
  - name: raw_url
    dtype: string
  - name: publication_date
    dtype: date32
  - name: article_id
    dtype: string
  splits:
  - name: '2017'
    num_bytes: 377124495
    num_examples: 103397
  - name: '2018'
    num_bytes: 1495305369
    num_examples: 461428
  - name: '2019'
    num_bytes: 2278782727
    num_examples: 593252
  - name: '2020'
    num_bytes: 4118175236
    num_examples: 1019049
  - name: '2021'
    num_bytes: 4096059437
    num_examples: 1035512
  - name: '2022'
    num_bytes: 4031480562
    num_examples: 994435
  download_size: 8846795459
  dataset_size: 16396927826
configs:
- config_name: default
  data_files:
  - split: '2017'
    path: data/2017-*
  - split: '2018'
    path: data/2018-*
  - split: '2019'
    path: data/2019-*
  - split: '2020'
    path: data/2020-*
  - split: '2021'
    path: data/2021-*
  - split: '2022'
    path: data/2022-*
extra_gated_description: >-
  This dataset is publicly accessible, but **you have to accept the conditions
  to access its files and content**.


  This dataset consists of articles from reliable and unreliable publishers.
  Many articles contain misinformation, hate speech, pseudo-scientific claims or
  otherwise toxic language.


  We release this dataset as a resource for researchers to investigate these
  phenomena and mitigate the harm of such language. We strongly urge you not to
  use this dataset for training publicly accessible language-models.


  We will not process or distribute your contact information.
extra_gated_prompt: >-
  Your request will be processed once you have accepted the
  licensing terms.
extra_gated_fields:
  I agree to attribute this work when distributing findings, models, datasets, etc, derived from this dataset: checkbox
  I agree to only use this dataset for non-commercial purposes: checkbox
  I agree to re-distribute altered versions of this dataset under the same Creative Commons license (CC BY-NC-SA 4): checkbox
  Intended use-case: text
---

# Dataset Card for `misinfo-general`

## Dataset Description
- **Homepage:** [https://github.com/ioverho/misinfo-general](https://github.com/ioverho/misinfo-general)
- **Paper:** [Yesterday’s News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models](https://arxiv.org/abs/2410.18122)
- **Point of Contact:** [i.o.verhoeven@uva.nl](mailto:i.o.verhoeven@uva.nl)

### Dataset Summary

We introduce `misinfo-general`, a benchmark dataset for evaluating misinformation models’ ability to perform out-of-distribution generalisation. Misinformation changes rapidly, much quicker than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation models need to be able to perform out-of-distribution generalisation, an understudied problem in existing datasets.

Constructed on top of the various NELA corpora ([2017](https://arxiv.org/abs/1803.10124), [2018](https://arxiv.org/abs/1904.01546), [2019](https://arxiv.org/abs/2003.08444), [2020](https://arxiv.org/abs/2102.04567), 2021, [2022](https://arxiv.org/abs/2203.05659)), `misinfo-general` is a large, diverse dataset consisting of news articles from reliable and unreliable publishers. Unlike NELA, we apply several rounds of deduplication and filtering to ensure all articles are of reasonable quality.

We use distant labelling to provide each publisher with rich metadata annotations. These annotations allow for simulating various generalisation splits that misinformation models are confronted with during deployment. We focus on 6 such splits-time, event, topic, publisher, political bias, misinformation type-but more are possible.

By releasing this dataset publicly, we hope to encourage future works that design misinformation models specifically with out-of-distribution generalisation in mind.

### Languages

The articles have been filtered to include only English texts. The publishers of the articles are primarily based in the continental USA, though some 'foreign' publishers are also present.

### How-to Use

This dataset comes in two parts. A set of `.parquet` files containing news articles in the `/data/` directory, and a publisher-level metadata database. The former is directly accessible using HuggingFace's datasets library, the latter requires `duckdb` to be installed.

If you only want to use the articles, without publisher metadata, first make sure you have access to this HuggingFace Hub repo. You can just use the [`datasets.load_dataset`](https://huggingface.co/docs/datasets/v3.0.1/en/package_reference/loading_methods#datasets.load_dataset):

```python
datasets.load_dataset(
  path="ioverho/misinfo-general",
  token=${YOUR_HUGGINGFACE_PA_TOKEN},
  **kwargs,
)
```

Make sure to replace `${YOUR_HUGGINGFACE_PA_TOKEN}` with your PA token. Check Profile > Settings > Access Tokens.

The recommended method for using this dataset, however, is by including the [metadata database](#metadata). Again, first make sure you have access this HuggingFace Hub repo. The easiest method for downloading everything together is by using HuggingFace's CLI:

```bash
huggingface-cli download --repo-type dataset --token ${YOUR_HUGGINGFACE_PA_TOKEN} --local-dir ./misinfo-general ioverho/misinfo-general
```

This should download the entire HuggingFace repository into a `misinfo-general` subdirectory of your current directory. Again, make sure to replace `${YOUR_HUGGINGFACE_PA_TOKEN}` with your PA token. Also make sure you are logged in to the HuggingFace CLI. You can check this by using the `huggingface-cli whoami` command.

Otherwise, if you have [git lfs](https://git-lfs.com/) isntalled, you can also `git clone`. This only works if your git ssh public key is registered with HuggingFace. See this [tutorial for more information](https://huggingface.co/docs/hub/en/security-git-ssh).

You can now use the dataset by calling:
```python
dataset = datasets.load_dataset(
    path="./misinfo-general",
)
```

You can now also access the publisher metadata database by calling:
```python
metadata_db = duckdb.connect(
    "./misinfo-general/metadata.db",
    read_only=True,
)
```

If you downloaded into a different local directory, make sure to alter the file path arguments.

## Dataset Structure

### Data Instances

The full dataset is structured as a `DatasetDict` with keys `["2017", ..., "2022"]` (these are strings). To get a specific instance, simply index one of the splits, e.g., `dataset["2020"][12345]`:

```json
{
    'source': 'themanchestereveningnews',
    'title': 'Video shows man hurling beer barrel through the front of Barclays Bank in Wigan town centre',
    'content': 'A man has been arrested after a beer barrel was hurled through the window of a bank in ... ',
    'author': 'newsdesk@men-news.co.uk (Sam Yarwood)',
    'domain': 'www.manchestereveningnews.co.uk',
    'raw_url': 'https://www.manchestereveningnews.co.uk/...',
    'publication_date': datetime.date(2020, 1, 10),
    'article_id': '2020-0012345'
}
```

### Data Fields

Each dataset year comes with the following columns:

| Field            | Type   | 2017 | 2018 | 2019 | 2020 | 2021 | 2022 |
| ---------------- | :----: | :--: | :--: | :--: | :--: | :--: | :--: |
| source           | str    | Y    | Y    | Y    | Y    | Y    | Y    |
| title            | str    | Y    | Y    | Y    | Y    | Y    | Y    |
| content          | str    | Y    | Y    | Y    | Y    | Y    | Y    |
| author           | str    | Y    |      | Y    | Y    | Y    | Y    |
| domain           | str    | Y    |      | Y    | Y    | Y    | Y    |
| raw_url          | str    | Y    |      | Y    | Y    | Y    | Y    |
| publication_date | date32 | Y    | Y    | Y    | Y    | Y    | Y    |
| article_id       | str    | Y    | Y    | Y    | Y    | Y    | Y    |

Note that 2018 is missing the `author`, `domain` and `raw_url` columns. To ensure the schema is complete, these columns only contain the "N/A" string.

### Publisher Metadata

The dataset comes with an accompanying metadata database, structured as a `duckdb=1.0.0` database. This format should be compatible with `duckdb>0.10.0` versions. We use this to make fast aggregation queries with minimal hassle. To start, connect to the database:

```python
import duckdb

metadata_db = duckdb.connect(
    "./misinfo-general/metadata.db",
    read_only=True
)
```

This can then be queried using SQL to find particular classes of articles. For example the following query, searches for all articles belonging to 'Left' or 'Extreme Left' publishers in 2020, sorted by their publication date:

```python
metadata_db.sql(
    """
    SELECT article_id, year, articles.source, title
    FROM articles INNER JOIN (
            SELECT DISTINCT source
            FROM sources
            WHERE (
                bias = 'Left'
                OR bias = 'Extreme Left'
            )
        ) AS pol_sources
        ON articles.source = pol_sources.source
    WHERE year = 2020
    ORDER BY article_id
    """
)
```

It should return an overview similar to the following print-out, which can be converted to a data loader over Python tuples or a `pandas` `dataframe`.

```txt
┌──────────────┬────────┬──────────────────┬───────────────────────────────────────────────────────────────────────────┐
│  article_id  │  year  │      source      │                                   title                                   │
│   varchar    │ uint16 │     varchar      │                                  varchar                                  │
├──────────────┼────────┼──────────────────┼───────────────────────────────────────────────────────────────────────────┤
│ 2020-0000006 │   2020 │ bipartisanreport │ EPA Scientists Break Ranks, Speak Out Against Trump Admin Dangers         │
│ 2020-0000009 │   2020 │ dailykos         │ January 1, 2030: The 2020s were CRAZY- Part One- POLITICS                 │
│ 2020-0000010 │   2020 │ dailykos         │ Happy New Year!!!: Rest. Then Resolve to Work Your Rumps Off - With a P…  │
│ 2020-0000016 │   2020 │ democracynow     │ First Lady of the World: Eleanor Roosevelt's Impact on New Deal to U.N.…  │
│ 2020-0000073 │   2020 │ thedailyrecord   │ Happy New Year from everyone at the Daily Record and Sunday Mail          │
│      ...     │   ...  │       ...        │                       ...                                                 │
```

To see how we use these to generate the above splits, check out our GitHub: [github](https://github.com/ioverho/misinfo-general). Also, make sure to check out the [`duckdb` documentation](https://duckdb.org/docs/api/python/overview).

### Publisher Metadata Schema

The following figure depicts the metadata database schema.

![A depiction of the metadata SQL schema.](/assets/db_schema.png)

Acronyms `PK` means primary key, a unique identifier for each row, whereas `FK` implies a foreign-key relationship, which links unique entities to rows in other tables. Each source has many articles in the `articles` table, and each topic also has many articles in the `articles` table. This is allows for rapid join operations to find articles belonging to a specific publisher or topic.

The `article_id` key is also present in the HuggngFace dataset, making it possible to link the dataset and the metadata database together.

### Data Splits

#### Years

This dataset contains articles collected from 2017 - 2022. Each year is used as a single split of the full dataset. To access a single year, simple index the produced `DatasetDict` instance using a *string* represenation of that year:
```python
dataset_2020 = datasets["2017"]
```

#### Generalisation Splitting

<!-- TODO: add paper url -->
To test for OoD generalisation, data splits should be generated using publisher level metadata, as discussed in our [paper](). This can be achieved using the provided metadata database.

In our paper, we generate 6 (+1 baseline) train/test splits of the dataset using the publisher-level metadata. Validation splits are sampled i.i.d. from the training set. In each split we try to ensure each split contains 70/10/20% of all articles. We define the following splits:

0. **Uniform**: standard stratified random splitting of articles into disjoint sets as a baseline
1. **Time**: train on a single dataset year, and evaluate on articles from seen publishers in all other years
2. **Event**: identify all articles which include keywords related to the COVID-19 pandemic, and reserve these as a held-out test set. We used a single event, but the dataset includes potentially thousands such events
3. **Topic**: we reserve the smallest topic clusters for the test set, and train on the rest
4. **Publisher**: we reserve the least frequent publishers for the test set, and train on the rest
5. **Political Bias**: we reserve all articles from either all leftist or rightist publishers for testing, and train on the opposing political bias along with centre biased ones. This is stratified over the different MBFC labels
6. **Misinformation Type**: similarly, we reserve all articles from either all 'Questionable Source' or 'Conspiracy-Pseudoscience' publishers for evaluation, and train on the opposing MBFC label, stratified over political bias. The reliable publishers are uniformly split across the datasets

Instead of cross-validation, we repeat each split for each year separately (6 times in total). While it might be possible to combine all years into a single dataset, the different years can be substantially different from each other.

### Special Tokens

We use 4 special tokens:
1. `<copyright>`: masks that occlude part of a sentence to ensure 'fair use'
2. `<twitter>`: masks twitter handles
3. `<url>`: masks URLs
4. `selfref`: masks references to the publisher that wrote the article

To ensure these tokens are used by your models, use HuggingFace's [`tokenizer.add_tokens`](https://huggingface.co/docs/transformers/en/main_classes/tokenizer#transformers.PreTrainedTokenizer.add_tokens) and [`model.resize_token_embeddings]([resize_token_embeddings](https://huggingface.co/docs/transformers/v4.45.2/en/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings)) functions.

## Dataset Creation

### Curation Rationale

The `misinfo-general` dataset came about to aggregate existing misinformation datasets for the specific purpose to testing out-of-distribution generalisation of misinformation classifiers. We start of with the various **NE**ws **LA**ndscape (NELA) corpora, which we filter, deduplicate, clean and standardize. We then completely updated the publisher level metadata, and clustered all articles based on topic.

### Source Data

#### Initial Data Collection and Normalization

All our articles come from the various **NE**ws **LA**ndscape (NELA) corpora. The NELA corpora cover 2017-2022 (6 iterations) almost continuously, with articles from a diverse group of publishers. In their original form, the 6 iterations together consist of 7.24 million long-form articles from hundreds of publishers. We filter, deduplicate, clean and standardize. We then completely updated the publisher level metadata, and clustered all articles based on topic. Ultimately, this left 4.16 million articles from 488 distinct publishers.

We downloaded the corpora from Harvard Dataverse, under a [CC0 1.0 licence](https://creativecommons.org/publicdomain/zero/1.0/deed.en). The corpora have since been de-accessioned.

#### Who are the source language producers?

The source language was scraped from various reliable and unreliable news publishers. Metadata on each article's publisher is included in the metadata database.

### Annotations

#### Annotation process

We use publisher-level annotations. This means we use a publisher's overall propensity for reliable reporting as a proxy label for an article's veracity. To use this dataset for supervised machine learning, the user is responsible for generation a mapping from publisher to reliability class.

#### Who are the annotators?

The publisher-level metadata is sourced from [Media Bias / Fact Check](https://mediabiasfactcheck.com/) (MBFC). MBFC is a curated database of news publishers, with thorough analyses of publisher origins, bias, and credibility. Despite being run by lay volunteers, prior work has found MBFC labels to correlate well with professional fact-checking sources. MBFC labels are dynamic and annotations can change, although this is infrequent. We use the latest available version of labels (Sept. 2024).

### Personal and Sensitive Information

We have removed identifiable twitter handles, urls and email strings. We have also removed references to the authoring publisher. Beyond that, all articles were scraped from publicly available articles.

## Considerations for Using the Data

### Social Impact of Dataset

The purpose of this dataset is to train and test for out-of-distribution generalisation of misinformation detection models. As such, it contains many unreliable publishers. These publishers can, and frequently do, produce false, hateful or toxic language. This is done in the form of clearly conspiratorial or pseudoscientific texts, but also in texts specifically written with the intent deceiving its audience. We **strongly** recommend derivatives of this dataset (e.g., language models fine-tuned on this data) not be shared openly.

### Discussion of Biases

While the dataset covers a large amount of publishers across a long time-span, the annotations come from a single source. This can lead to biases in the publisher-level metadata provided. In general, the MBFC judgements represent information about publishers from a narrow, US-centric point of view. Labels derived from these annotations likely do not extend to labels produced by individuals from different cultural or socio-economic backgrounds.

### Other Known Limitations

[More Information Needed]

## Additional Information

### Dataset Curators

This dataset was initially created by Ivo Verhoeven, supervised by Pushkar Mishra and Katia Shutova. The original NELA corpora were put together by Benjamin Horne, William Dron, Sara Khedr, Sibel Adalı, Jeppe Norregaard and Maurício Gruppi.

### Licensing Information

[![License: CC BY-NC-ND 4.0](https://licensebuttons.net/l/by-nc-nd/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-nd/4.0/)

We licensed this dataset under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en). This allows users to copy, adapt and redistribute the data as long as the data is attributed, shared under the same license and used for non-commercial purposes. To ensure responsible usage, we have also decided to limit initally review access requests.

### Citation Information

```bibtex
@misc{verhoeven2024yesterdaysnewsbenchmarkingmultidimensional,
      title={Yesterday's News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models},
      author={Ivo Verhoeven and Pushkar Mishra and Ekaterina Shutova},
      year={2024},
      eprint={2410.18122},
      archivePrefix={arXiv},
      primaryClass={cs.IR},
      url={https://arxiv.org/abs/2410.18122},
}
```
