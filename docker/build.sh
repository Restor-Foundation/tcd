#!/bin/bash

rm -rf ./docker-tcd-pipeline
REPO=$(awk '/url/{print $3}' ../.git/config)
echo $REPO
git clone $REPO docker-tcd-pipeline --depth 1
rm -rf docker-tcd-pipeline/.git
docker build -t tcd . -f $1
rm -rf ./docker-tcd-pipeline