#!/bin/bash

rm -rf ./docker-tcd-pipeline
git clone https://github.com/Restor-Foundation/tcd-pipeline.git docker-tcd-pipeline --depth 1 --single-branch --branch post_processing_refactor
# git clone git@github.com:Restor-Foundation/tcd-pipeline.git docker-tcd-pipeline --depth 1 --single-branch --branch post_processing_refactor
rm -rf docker-tcd-pipeline/.git
cp -r ../checkpoints ./checkpoints
docker build -t tcd . -f $1
rm -rf ./docker-tcd-pipeline
rm -rf ./checkpoints