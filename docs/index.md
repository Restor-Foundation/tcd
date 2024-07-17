# TCD Pipeline Documentation

![Coverage Status](coverage-badge.svg)
[![Tests](https://github.com/Restor-Foundation/tcd/actions/workflows/python-test.yml/badge.svg)](https://github.com/Restor-Foundation/tcd/actions/workflows/python-test.yml)
[![Docker](https://github.com/Restor-Foundation/tcd/actions/workflows/docker.yml/badge.svg)](https://github.com/Restor-Foundation/tcd/actions/workflows/docker.yml)
[![arXiv](https://img.shields.io/badge/arXiv-2407.11743-b31b1b.svg)](https://arxiv.org/abs/2407.11743)
[![Documentation](https://img.shields.io/badge/Documentation-blue.svg)](https://restor-foundation.github.io/tcd/)


Researchers at Restor and ETH Zurich have developed this powerful and easy-to-use pipeline for detecting individual trees and tree canopy in aerial images. You can also explore our open-access dataset of labelled trees which may be useful for training your own models or benchmarking against other methods.

![Zurich tree map, monochrome](images/zurich_trees_mono.jpg)
_Trees detected from aerial imagery over the city of Zurich_

Head over to the [introduction](introduction.md) page for general inforamtion about the project.

For more information about using our models and pipeline:

- [How to install the pipeline](install.md)
- [Predicting tree cover in images](prediction.md)
- [Training your own models](training.md)
- [Exporting models for production deployment](deployment.md)
- [Sample model card](model.md)
- [Datasets and data formats](datasets.md)
- [Benchmarking](benchmark.md)
- [Output caching](cache.md)
- [Pipeline architecture](architecture.md)
- [API/developer reference](reference.md)

## Quickstart

This quickstart assumes that you have Conda installed. Open a terminal and

First, clone the repository:

```bash
git clone github.com/restor-foundation/tcd
cd tcd
```

Then, install the conda environment and install the pipeline:

```bash
conda env create -f environment.yml
pip install -e .[test]
```

And run a test prediction on sample data in the repo:

```bash
tcd-predict semantic input=data/5c15321f63d9810007f8b06f_10_00000.tif output=test_prediction
```

## Citation

```bibtex
@misc{oamtcdpreprint,
      title={OAM-TCD: A globally diverse dataset of high-resolution tree cover maps}, 
      author={Josh Veitch-Michaelis and Andrew Cottam and Daniella Schweizer and Eben N. Broadbent and David Dao and Ce Zhang and Angelica Almeyda Zambrano and Simeon Max},
      year={2024},
      eprint={2407.11743},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2407.11743}, 
}
```