#!/bin/bash

models = ( 'kmeans_3' 'kmeans_5' 'kmeans_10' 'gdumb_3' 'gdumb_5' 'gdumb_10' )

for model in "${models[@]}"
do
    python3 eval/batch_eval_celeb.py
done