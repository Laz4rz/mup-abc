#!/bin/bash

echo "Running MLP experiment..."
python3 train_mlp.py --model MLP --subset 0.3

echo "Running muMLP experiment..."
python3 train_mlp.py --model muMLP --subset 0.3

echo "All experiments completed."
