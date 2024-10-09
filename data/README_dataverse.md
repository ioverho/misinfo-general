# Dataset Card for `misinfo-general`

## Dataset Description
<!-- TODO: add paper url -->
- **Homepage:** [https://github.com/ioverho/misinfo-general](https://github.com/ioverho/misinfo-general)
- **Paper:** [Yesterday’s News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models]()
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

First make sure you have access to the dataset files. Download the entire repository, and unzip.

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

![A depiction of the metadata SQL schema.](../assets/db_schema.png)

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

We licensed this dataset under [CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en). This allows users to copy, adapt and redistribute the data as long as the data is attributed, shared under the same license and used for non-commercial purposes. To ensure responsible usage, we have also decided to limit initally review access requests.

### Citation Information

```bibtex
@article{TODO: finish paper citation
  title="Yesterday’s News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models",
  author = {Verhoeven, Ivo and Mishra, Pushkar and Shutova, Ekaterina},
  year = {2024},
  month = oct,
}
```
