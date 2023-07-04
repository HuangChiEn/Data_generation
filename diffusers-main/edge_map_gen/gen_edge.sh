#!/bin/bash

edge_type='coarse'

if [ $edge_type = 'finer' ] ; then
    python gen_edge.py \
        --dataset cityscapes --datadir="/data1/dataset/Cityscapes" --outdir="./"
elif [ $edge_type = 'coarse' ] ; then 
    python gen_coarse_edge.py \
        --dataset cityscapes --datadir="/data1/dataset/Cityscapes" --outdir="./"
else 
    printf "invalid argument %s\n" $edge_type
    exit 1
fi