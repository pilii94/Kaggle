import argparse
import datetime
import glob
import json
import logging
import numpy as np
import os
import re
import pickle
import random
import sys
import time
import traceback
import pandas as pd
from collections import defaultdict
from functools import partial
sys.path.append('..')

# Define visible CUDA DEVICES
import os

# from spacy.scorer import Scorer
# from spacy.util import decaying

import spacy
from spacy.util import minibatch, compounding
from spacy.gold import GoldParse

spacy.prefer_gpu()  # For gpu usage if possible. Better
# spacy.require_gpu()  # For gpu usage or error

from tqdm import tqdm
from termcolor import colored
from pathlib import Path

from sklearn.metrics import accuracy_score, f1_score, precision_recall_fscore_support
sys.path.append('../')
from common.Bert_Finetune import BertClassifier
from src.train_config import config

def train_models(config = config, logger = None):
    
    data_train = pd.read_csv(config['train_data'])
    data_train.reset_index(drop=True)

    out_path = os.path.join(config['output_dir'], config['model_name'],'v1')
    model = BertClassifier(logger)
    model.fit(data_train["excerpt"],data_train["target"])
    model.save(out_path)  




if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('train_models')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(colored('[%(asctime)s]', 'magenta') +
                                  colored('[%(levelname)s] ',
                                          'blue') + '%(message)s',
                                  '%Y-%m-%d %H:%M:%S')

    logging_file_handler = logging.FileHandler(
        f'train_models_{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging_file_handler.setLevel(logging.DEBUG)
    logging_file_handler.setFormatter(formatter)
    logger.addHandler(logging_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f'Starting model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
    train_start_time = time.time()


    logger.info('Working directory: ' + colored(f'{config["output_dir"]}', 'green'))

    train_models(config=config, logger=logger)

    train_end_time = time.time()
    train_time = train_end_time - train_start_time
    logger.info(f'Train time: {train_time:.2} s.')
    logger.info(f'Done model training PID: ' +
                colored(f'{os.getpid()}', 'green'))
