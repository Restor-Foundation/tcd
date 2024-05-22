#!/bin/bash

rm -rf ./docker-tcd-pipeline
git clone git@github.com:jveitchmichaelis/tcd.git docker-tcd-pipeline --depth 1
rm -rf docker-tcd-pipeline/.git
cp -r ../checkpoints ./checkpoints
docker build -t tcd . -f $1
rm -rf ./docker-tcd-pipeline
rm -rf ./checkpoints
