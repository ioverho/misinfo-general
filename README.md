<h1 align="center">Yesterday’s News: Benchmarking Multi-Dimensional Out-of-Distribution Generalisation of Misinformation Detection Models
</h1>

<!-- TODO add paper URL -->

<!-- TODO add URL to ACL anthology -->


<p align="center"><b>Ivo Verhoeven<sup>&dagger;</sup>, Pushkar Mishra<sup>&Dagger;</sup> and Ekaterina Shutova<sup>&dagger;</sup></b></p>

<p align="center"><small>&dagger; ILLC, University of Amsterdam, &Dagger; MetaAI, London</small></p>

<!-- <p align="center">
<a href="https://arxiv.org/abs/2404.01822">
    <img src="https://img.shields.io/static/v1.svg?logo=arxiv&label=Paper&message=Open%20Paper&color=green"
    alt="Arxiv Link"
    style="float: center;"
    />
</a>
</p> -->

<p align="center">
<a href="https://huggingface.co/datasets/ioverho/misinfo-general"><img src="./assets/hf-badge-hf.svg" alt="Dataset on HF"></a>
<a href="https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TXXUFN"><img src="./assets/hf-badge-dataverse.svg" alt="Dataset on Harvard Dataverse"></a>
</p>



This GitHub repository contains documentation for `misinfo-general`, and code used for our accompanying paper. With it we hope to introduce new data and evaluation methods for testing and training for out-of-distribution of generalisation in misinformation detection models.

Please direct your questions to: [i.o.verhoeven@uva.nl](mailto:i.o.verhoeven@uva.nl)

## Abstract

> This paper introduces `misinfo-general`, a benchmark dataset for evaluating misinformation models’ ability to perform out-of-distribution generalisation. Misinformation changes rapidly, much quicker than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation models need to be able to perform out-of-distribution generalisation, an understudied problem in existing datasets. We identify 6 axes of generalisation—time, event, topic, publisher, political bias, misinformation type—and design evaluation procedures for each. We also analyse some baseline models, highlighting how these fail important desiderata.

## Structure

```txt
/config/
    various configuration YAML files
/data/
    ├── README_dataverse.md
    │       the dataset card used for storing data on Harvard Dataverse
    └── README_dataverse.md
            the dataset card used for storing data on Hugging Face Hub
/scripts/
    various scripts for running various experiments on a SLURM cluster
/src/
    ├── /misinfo_general/
    │       utility code
    └── *.py
            top level scripts for training and evaluating misinformation models on misinfo-general
/env.yaml/
    conda environment used for local development
/env_snellius.yaml/
    conda environment used for training and evaluation on a SLURM cluster
```

## Data

We have released our data on two separate platforms: [Hugging Face Hub](https://huggingface.co/datasets/ioverho/misinfo-general) and [Harvard Dataverse](https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/TXXUFN). Both of these repositories require access requests before downloading is possible. We provide additional detail on their respective [dataset cards](./data/).

The dataset is licensed under [CC BY-SA-NC 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en). This allows for sharing and redistribution, but requires attribution and sharing derivatives under similar terms. It does permit commercial use-cases.

On either repo, we provide data in a set of `.arrow` files, which can be read using a variety of packages although we used `datasets`, an provide the publisher-level metadata in a `duckdb` database. Upon request, we can change the formatting of either the dataset or metadata database.

### Content

Because of the nature of the language it includes, `misinfo-general` contains texts that are toxic, hateful, or otherwise harmful to society if disseminated. The dataset itself or any derivative formats of it, like LLMs, should not be released for non-research purposes. The texts themselves might also be copyrighted by their original publishers.

We have deliberately removed all social media content, and all hyperlinks to such content. We consider such content Personally identifiable information (PII), with limited use in misinformation classification beyond author profiling. Such applications are fraught with ethical problems, and likely only induce overfitting in text-based classification.

## Code & Environment

The development environment is stored as a `conda` readable YAMl file in [`./env.yaml`](./env.yaml). The training environment, used on the Snellius supercomputer, can be found in [`./env_snellius.yaml`](./env_snellius.yaml).

For configuration, we used [Hydra](https://hydra.cc/docs/intro/). The configuration files may be fund in [`./config`](./config/). All scripts in `/main/` can be run from the command line, using the Hydra syntax. For example,

```bash
python src/train_uniform.py \
    fold=0 \
    year=2017 \
    seed=942 \
    model_name='microsoft/deberta-v3-base' \
    data.max_length=512 \
    batch_size.tokenization=1024 \
    batch_size.train=64 \
    batch_size.eval=128 \
    ++trainer.kwargs.fp16=true \
    ++trainer.kwargs.use_cpu=false \
    ++trainer.memory_metrics=false \
    ++trainer.torch_compile=false \
    ++optim.patience=5 \
    data_dir=$LOCAL_DATA_DIR \
    disable_progress_bar=true
```

uses the data stored at `$LOCAL_DATA_DIR` to train a uniform split model on the 2017 data iteration.

## Citation

<!-- TODO add paper URL to citation-->

```bibtex
Add example citation
```
