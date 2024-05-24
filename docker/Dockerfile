FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-runtime
ENV TZ=Europe/Zurich

# set bash as current shell
RUN chsh -s /bin/bash
SHELL ["/bin/bash", "-c"]

# Apt update and clean
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update
RUN apt-get install -y wget bzip2 build-essential ca-certificates libglib2.0-0 libxext6 libsm6 libxrender1 git mercurial subversion && \
        apt-get clean

# setup Restor user
RUN groupadd -r restor && useradd -r -g restor restor
RUN mkdir /home/restor
RUN chown -R restor:restor /home/restor
RUN chmod 755 /home/restor

USER restor
WORKDir /home/restor

ADD docker-tcd-pipeline tcd-pipeline

USER root
RUN chown -R restor:restor /home/restor/tcd-pipeline
USER restor

# # test suite
WORKDIR tcd-pipeline
RUN python -m pip install -r requirements.txt
RUN python -m pip install pytest pytest-cov
RUN python -m pip install -e .
RUN echo "conda activate tcd" >> ~/.bashrc

#RUN python -m pytest -sx --verbose --trace
