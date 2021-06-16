#!/usr/bin/env python
import datetime

config = {}

config['epochs'] = 500 # Training iterations

config['output_dir'] = f'../models/'  # Directory where all results are saved

config['train_size'] = 0.8

config['train_data'] = "../../data/train.csv"
config['test_data'] = "../../data/test.csv"
config['submission'] = "../../data/sample_submission.csv"


config['model_name'] = "bert-base-uncased"
config['MAX_LEN'] = 512
config['lr'] = 2e-5
config['eps'] = 1e-8 
config['batch_size'] = 10

