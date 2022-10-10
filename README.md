# tcd-pipeline

Pre-install requirements:

* Create a conda env e.g. `conda create -n tcd python=3.10`
* Install torch `conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch -c nvidia`
* Install requirements `pip install -r requirements.txt` (this should pick up detectron)
* Install git LFS `sudo apt install git-lfs` and pull the checkpoint `git lfs pull`
