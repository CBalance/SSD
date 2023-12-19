#!/bin/bash

T=`date +%m%d%H%M%S`

mkdir exp
mkdir exp/$T
mkdir exp/$T/code
cp -r datasets exp/$T/code/datasets
cp -r models exp/$T/code/models
cp ./*.py exp/$T/code/
cp run.sh exp/$T/code

mkdir exp/$T/train.log

datapath=

python main.py --data-path $datapath --batch-size 4 --accumulation-steps 1 --tag $T 2>&1 | tee exp/$T/train.log/running.log
