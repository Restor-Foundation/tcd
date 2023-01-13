FROM nvidia/cuda:12.0.0-devel-ubuntu20.04
ENV TZ=Europe/Zurich

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Apt update and clean
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget bzip2 build-essential ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

# Get Miniconda
RUN wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
        /bin/bash ~/miniconda.sh -b -p /opt/conda && \
        rm ~/miniconda.sh && \
        ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh && \
        find /opt/conda/ -follow -type f -name '*.a' -delete && \
        find /opt/conda/ -follow -type f -name '*.js.map' -delete && \
        /opt/conda/bin/conda clean -afy && \
        /opt/conda/bin/conda update conda && \
        /opt/conda/bin/conda init

# set path to conda
ENV PATH /opt/conda/bin:$PATH

# setup Restor user
RUN groupadd -r restor && useradd -r -g restor restor
RUN mkdir /home/restor
RUN chown -R restor:restor /home/restor
RUN chmod 755 /home/restor

USER restor
RUN echo ". /opt/conda/etc/profile.d/conda.sh" >> ~/.bashrc
WORKDir /home/restor

# setup conda virtual environment
ADD environment.yaml /tmp/environment.yaml
RUN conda init
RUN conda env create -f /tmp/environment.yaml
ENV CONDA_DEFAULT_ENV tcd

ADD docker-tcd-pipeline tcd-pipeline

USER root
RUN chown -R restor:restor /home/restor/tcd-pipeline
USER restor

# # test suite
WORKDIR tcd-pipeline
SHELL ["conda", "run", "-p", "/home/restor/.conda/envs/tcd", "/bin/bash", "-c"]
RUN python -m pip install -r requirements.txt
RUN python -m pip install pytest pytest-cov
RUN python -m pip install torchmetrics==0.10.3
RUN python -m pip install -e .
RUN echo "conda activate tcd" >> ~/.bashrc
ADD checkpoints checkpoints
#RUN python -m pytest test/unit