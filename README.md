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

This GitHub repository contains code used for our accompanying paper. We introduce a new benchmark dataset, `misinfo-general`, for testing and training for out-of-distribution of generalisation in misinformation detection models.

Please direct your questions to: [i.o.verhoeven@uva.nl](mailto:i.o.verhoeven@uva.nl)

## Abstract

> This paper introduces misinfo-general, a benchmark dataset for evaluating misinformation models’ ability to perform out-of-distribution generalisation. Misinformation changes rapidly, much quicker than moderators can annotate at scale, resulting in a shift between the training and inference data distributions. As a result, misinformation models need to be able to perform out-of-distribution generalisation, an understudied problem in existing datasets. We identify 6 axes of generalisation—time, event, topic, publisher, political bias, misinformation type—and design evaluation procedures for each. We also analyse some baseline models, highlighting how these fail important desiderata.

## Structure

```txt
/config/
    various configuration YAML files
/scripts/
    various scripts for running various experiments on a SLURM cluster
/src/
    ├── /misinfo_benchmark_models/
    │       utility code
    └── *.py
            top level scripts for training and evaluating misinformation models on misinfo-general
/env.yaml/
    conda environment used for local development
/env_snellius.yaml/
    conda environment used for training and evaluation on a SLURM cluster
```

## Data

We will be releasing our datasets publicly soon. These will be available on both the Harvard Dataverse and the HuggingFace Hub. Either variant will require requesting access, to ensure responsible usage.

For more detail, please read the [dataset card](./DATASET_CARD.MD).

The datasets are licensed under [CC BY-SA-NC](https://creativecommons.org/licenses/by-nc-sa/4.0/deed.en). This allows for sharing and redistribution, but requires attribution and sharing derivatives under similar terms. It does permit commercial use-cases.

### Content

By its very nature, `misinfo-general` contains texts that are toxic, hateful, or otherwise harmful to society if disseminated. The dataset itself or any derivative formats of it, like LLMs, should not be released for non-research purposes. The texts themselves might also be copyrighted by their original publishers.

We have deliberately removed all social media content, and all hyperlinks to such content. We consider such content Personally identifiable information (PII), with limited use in misinformation classification beyond author profiling. Such applications are fraught with ethical problems, and likely only induce overfitting in text-based classification.

## Citation

<!-- TODO add paper URL to citation-->

```bibtex
Add example citation
```