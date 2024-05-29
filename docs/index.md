# Index

Head over to the [introduction](introduction.md) page for general inforamtion about the project.

For more information about using our models and pipeline:

- [How to install the pipeline](install.md)
- [Predicting tree cover in images](prediction.md)
- [Training your own models](training.md)
- [Exporting models for production deployment](deployment.md)
- [Datasets and data formats](datasets.md)
- [Pipeline architecture](architecture.md)
- [API/developer reference](reference.md)

## Quickstart

This quickstart assume that you have Conda installed. Open a terminal and

First, clone the repository:

```bash
git clone github.com/jveitchmichaelis/tcd
cd tcd
```

Then, install the conda environment and install the pipeline:

```bash
conda env create -f environment.yml
pip install -e .[test]
```

And run a test prediction on sample data in the repo:

```bash
python predict.py semantic input=data/5c15321f63d9810007f8b06f_10_00000.tif output=test_prediction
```
