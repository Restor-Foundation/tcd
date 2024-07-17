#!/bin/bash

rm -rf ./docker-tcd
REPO=$(awk '/url/{print $3}' ../.git/config)
echo $REPO
git clone $REPO docker-tcd --depth 1
rm -rf docker-tcd/.git
docker build -t tcd . -f $1
rm -rf ./docker-tcd