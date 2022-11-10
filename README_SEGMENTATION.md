# Restor Foundation Tree Crown Delineation Pipeline - Segmantation

This README file contains information to set up the environment and run training for the segmentation model.

1. Follow the steps in the main README file to set up the basics dependencies

2. Install the additional requirements with from pip:

    ```bash
    pip install -r requirements_segmentation.txt
    ```
    Note that the `requirements_segmentation.txt` file is generated with `pipreqs` to ensure that only requirements that are imported and used are stored in the requirements file, avoiding all writing all the requirements installed via pip.
    
3. Set up W&B as described here: https://docs.wandb.ai/quickstart

4. `cd` into src and run `masking.py` to generate the masks from the given data

5. `cd` back to the root folder and then to utils and run `clean_data.py` to fix the problem with BW images

6. `cd` back to the root folder of the repositry and create a file named `.env` containing the following information
    ```
    DATA_DIR = /direcotry/to/data
    LOG_DIR = /direcotry/to/logs
    REPO_DIR = /directory/of/the/project
    ```
    Note that `DATA_DIR` and `LOG_DIR` should be in some place that allows for storage of large quantities of data (i.e., not the `home` folder of Euler). These are stored as environment variables as they might differ for each person.
    
7. Run the training by:
    ```bash
    python vanilla_model.py
    ```
    The configuration used for running the training is contained in `conf.yaml` and the configuration for running sweeps on W&B is in `conf_sweep.yaml`.
    If running this on Euler, one needs to specify how much memory and the GPU required (the memory of the GPU should be >= 20GB).
