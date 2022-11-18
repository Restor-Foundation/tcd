# Restor Foundation Tree Crown Delineation Pipeline - Segmentation

This README file contains information to set up the environment and run training for the segmentation model.

1. Follow the steps in the main README file to set up the basics dependencies

2. Install the additional requirements with from pip:

    ```bash
    pip install -r requirements_segmentation.txt
    ```
    Note that the `requirements_segmentation.txt` file is generated with `pipreqs` to ensure that only requirements that are imported and used are stored in the requirements file, avoiding all writing all the requirements installed via pip.
    
3. Set up W&B as described here: https://docs.wandb.ai/quickstart

4. Generate the masks from the dataset splits:

    ```bash

    python src/masking.py --images ./data/restor-tcd-oam/images --annotations ./data/restor-tcd-oam/train_20221010.json --prefix train
    python src/masking.py --images ./data/restor-tcd-oam/images --annotations ./data/restor-tcd-oam/val_20221010.json --prefix val
    python src/masking.py --images ./data/restor-tcd-oam/images --annotations ./data/restor-tcd-oam/test_20221010.json --prefix test

    ```

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

* Weights and Biases (WandB)
When training with WandB, the project name should be indicated in the `conf.yaml` file. (In future versions a flag will be added to choose whether to upload the training metrics and logs to WandB).
* Sweep configuration: hyperparameter search in WandB
To run sweeping for changing hyperparameters, these steps should be followed:
1. Set the parameter sweep in `conf.yaml` to True and change the project name to `vanilla-model-sweep-runs`
2. Access the `conf_sweep.yaml` file and add/remove/change the hyperparameters observed there. For now, loss, segmentation_model and backbone are considered. They should be changed in the list as new items. The method for sweeping can also be changed from grid (all possible combinations) to random.
3. Run the training as before.