#!/bin/bash

rm -rf ./docker-tcd-pipeline
git clone git@github.com:Restor-Foundation/tcd-pipeline.git docker-tcd-pipeline --depth 1 --single-branch --branch post-processing-refactor
rm -rf docker-tcd-pipeline/.git
cp -r ../checkpoints ./checkpoints
docker build -t tcd .
rm -rf ./docker-tcd-pipeline
rm -rf ./checkpoints
