import pandas as pd
import numpy as np
import datetime
import matplotlib
from matplotlib import pyplot as plt
import time
import random
import logging
import argparse
from termcolor import colored
import os
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.metrics import f1_score
import gc
from collections import defaultdict
from config import config
# nlp
from sklearn.feature_extraction.text import CountVectorizer
import torch
import torch.nn as nn
from transformers import LongformerConfig, LongformerModel, LongformerTokenizerFast
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import sys

sys.path.insert(0,"../common/")
from utils import active_logits, active_labels, active_preds_prob, EarlyStopping



def agg_essays(train_flg):
    folder = 'train' if train_flg else 'test'
    names, texts =[], []
    for f in tqdm(list(os.listdir(f'{config["data_dir"]}/{folder}'))):
        names.append(f.replace('.txt', ''))
        texts.append(open(f'{config["data_dir"]}/{folder}/' + f, 'r').read())
        df_texts = pd.DataFrame({'id': names, 'text': texts})

    df_texts['text_split'] = df_texts.text.str.split()
    print('Completed tokenizing texts.')
    return df_texts

def ner(df_texts, df_train):
    all_entities = []
    for _,  row in tqdm(df_texts.iterrows(), total=len(df_texts)):
        total = len(row['text_split'])
        entities = ['0'] * total

        for _, row2 in df_train[df_train['id'] == row['id']].iterrows():
            discourse = row2['discourse_type']
            list_ix = [int(x) for x in row2['predictionstring'].split(' ')]
            entities[list_ix[0]] = f'B-{discourse}'
            for k in list_ix[1:]: entities[k] = f'I-{discourse}'
        all_entities.append(entities)

    df_texts['entities'] = all_entities
    print('Completed mapping discourse to each token.')
    return df_texts

def preprocess(df_train = None):
    if df_train is None:
        train_flg = False
    else:
        train_flg = True
    
    df_texts = agg_essays(train_flg)

    if train_flg:
        df_texts = ner(df_texts, df_train)
    return df_texts

def seed_everything(seed=config['random_seed']):
    #os.environ['PYTHONSEED'] = str(seed)
    np.random.seed(seed%(2**32-1))
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic =True
    torch.backends.cudnn.benchmark = False

def split_fold(df_train):
    ids = df_train['id'].unique()
    kf = KFold(n_splits=config['n_fold'], shuffle = True, random_state=config['random_seed'])
    for i_fold, (_, valid_index) in enumerate(kf.split(ids)):
        df_train.loc[valid_index,'fold'] = i_fold
    return df_train

class FeedbackPrizeDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len, has_labels):
        self.len = len(dataframe)
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.has_labels = has_labels
    
    def __getitem__(self, index):
        text = self.data.text[index]
        encoding = self.tokenizer(
            text.split(),
            is_split_into_words = True,
            padding = 'max_length',
            truncation = True,
            max_length = self.max_len
        )
        word_ids = encoding.word_ids()

        # targets
        if self.has_labels:
            word_labels = self.data.entities[index]
            prev_word_idx = None
            labels_ids = []
            for word_idx in word_ids:
                if word_idx is None:
                    labels_ids.append(config['IGNORE_INDEX'])
                elif word_idx != prev_word_idx:
                    labels_ids.append(config['LABELS_TO_IDS'][word_labels[word_idx]])
                else:
                    if config['label_subtokens']:
                        labels_ids.append(config['LABELS_TO_IDS'][word_labels[word_idx]])
                    else:
                        labels_ids.append(config['IGNORE_INDEX'])
                prev_word_idx = word_idx
            encoding['labels'] = labels_ids
        # convert to torch.tensor
        item = {k: torch.as_tensor(v) for k, v in encoding.items()}
        word_ids2 = [w if w is not None else config['NON_LABEL'] for w in word_ids]
        item['word_ids'] = torch.as_tensor(word_ids2)
        return item

    def __len__(self):
        return self.len

class FeedbackModel(nn.Module):
    def __init__(self):
        super(FeedbackModel, self).__init__()
        model_config = LongformerConfig.from_pretrained(config['model_name'])
        self.model_config = model_config
        self.backbone = LongformerModel.from_pretrained(config['model_name'], config=model_config)
        self.head = nn.Linear(model_config.hidden_size, config['num_labels'])
    
    def forward(self, input_ids, mask):
        x = self.backbone(input_ids, mask)
        logits = self.head(x[0])
        return logits



def calc_overlap(row):
    """
    calculate the overlap between prediction and ground truth
    """
    set_pred = set(row.new_predictionstring_pred.split(' '))
    set_gt = set(row.new_predictionstring_gt.split(' '))
    # length of each end intersection
    len_pred = len(set_pred)
    len_gt = len(set_gt)
    intersection = len(set_gt.intersection(set_pred))
    overlap_1 = intersection / len_gt
    overlap_2 = intersection / len_pred
    return [overlap_1, overlap_2]

def score_feedback_comp(pred_df, gt_df):
    """
    A function that scores for the kaggle
        Student Writing Competition
        
    Uses the steps in the evaluation page here:
        https://www.kaggle.com/c/feedback-prize-2021/overview/evaluation
    """
    gt_df = gt_df[['id', 'discourse_type', 'new_predictionstring']].reset_index(drop = True).copy()
    pred_df = pred_df[['id', 'class', 'new_predictionstring']].reset_index(drop = True).copy()
    gt_df['gt_id'] = gt_df.index
    pred_df['pred_id'] = pred_df.index
    joined = pred_df.merge(
        gt_df,
        left_on = ['id', 'class'],
        right_on = ['id', 'discourse_type'],
        how = 'outer',
        suffixes = ['_pred', '_gt']
    )
    joined['new_predictionstring_gt'] =  joined['new_predictionstring_gt'].fillna(' ')
    joined['new_predictionstring_pred'] =  joined['new_predictionstring_pred'].fillna(' ')
    joined['overlaps'] = joined.apply(calc_overlap, axis = 1)
    # overlap over 0.5: true positive
    # If nultiple overlaps exists, the higher is taken.
    joined['overlap1'] = joined['overlaps'].apply(lambda x: eval(str(x))[0])
    joined['overlap2'] = joined['overlaps'].apply(lambda x: eval(str(x))[1])

    joined['potential_TP'] = (joined['overlap1'] >= 0.5) & (joined['overlap2'] >= 0.5)
    joined['max_overlap'] = joined[['overlap1', 'overlap2']].max(axis = 1)
    tp_pred_ids = joined.query('potential_TP').sort_values('max_overlap', ascending = False)\
                  .groupby(['id', 'new_predictionstring_gt']).first()['pred_id'].values
    
    fp_pred_ids = [p for p in joined['pred_id'].unique() if p not in tp_pred_ids]
    matched_gt_ids = joined.query('potential_TP')['gt_id'].unique()
    unmatched_gt_ids = [c for c in joined['gt_id'].unique() if c not in matched_gt_ids]

    TP = len(tp_pred_ids)
    FP = len(fp_pred_ids)
    FN = len(unmatched_gt_ids)
    macro_f1_score = TP / (TP + 1/2 * (FP + FN))
    return macro_f1_score

def oof_score(df_val, oof):
    f1score = []
    classes = ['Lead', 'Position', 'Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val.loc[df_val['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f'{c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    return f1avg

def inference(model, dl, criterion, valid_flg):
    final_predictions = []
    final_predictions_prob = []
    stream = tqdm(dl)
    model.eval()
    
    valid_loss = 0
    valid_f1sc = 0
    all_logits = None
    for batch_idx, batch in enumerate(stream, start = 1):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        with torch.no_grad():
            raw_logits = model(input_ids=ids, mask = mask)
        del ids, mask
        
        word_ids = batch['word_ids'].to(device, dtype = torch.long)
        if valid_flg:    
            raw_labels = batch['labels'].to(device, dtype = torch.long)
            logits = active_logits(raw_logits, word_ids)
            labels = active_labels(raw_labels)
            preds, preds_prob = active_preds_prob(logits)
            valid_f1sc += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average= 'macro')
            loss = criterion(logits, labels)
            valid_loss += loss.item()
        
        if batch_idx == 1:
            all_logits = raw_logits.cpu().numpy()
        else:
            all_logits = np.append(all_logits, raw_logits.cpu().numpy(), axis=0)

    
    if valid_flg:        
        epoch_loss = valid_loss / batch_idx
        epoch_f1sc = valid_f1sc / batch_idx
    else:
        epoch_loss, epoch_f1sc = 0, 0
    return all_logits, epoch_loss, epoch_f1sc


def preds_class_prob(all_logits, dl):
    print("predict target class and its probabilty")
    final_predictions = []
    final_predictions_score = []
    stream = tqdm(dl)
    len_sample = all_logits.shape[0]

    for batch_idx, batch in enumerate(stream, start=0):
        for minibatch_idx in range(config['valid_batch_size']):
            sample_idx = int(batch_idx * config['valid_batch_size'] + minibatch_idx)
            if sample_idx > len_sample - 1 : break
            word_ids = batch['word_ids'][minibatch_idx].numpy()
            predictions =[]
            predictions_prob = []
            pred_class_id = np.argmax(all_logits[sample_idx], axis=1)
            pred_score = np.max(all_logits[sample_idx], axis=1)
            pred_class_labels = [config['IDS_TO_LABELS'][i] for i in pred_class_id]
            prev_word_idx = -1
            for idx, word_idx in enumerate(word_ids):
                if word_idx == -1:
                    pass
                elif word_idx != prev_word_idx:
                    predictions.append(pred_class_labels[idx])
                    predictions_prob.append(pred_score[idx])
                    prev_word_idx = word_idx
            final_predictions.append(predictions)
            final_predictions_score.append(predictions_prob)
    return final_predictions, final_predictions_score


def get_preds_onefold(model, df, dl, criterion, valid_flg):
    logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg)
    all_preds, all_preds_prob = preds_class_prob(logits, dl)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred, valid_loss, valid_acc

def get_preds_folds(df, dl, criterion, valid_flg=False):
    for i_fold in range(config['n_fold']):
        model_filename = os.path.join(config['model_dir'], f"{config['model_savename']}_{i_fold}.bin")
        print(f"{model_filename} inference")
        model = FeedbackModel()
        model = model.to(device)
        model.load_state_dict(torch.load(model_filename))
        logits, valid_loss, valid_acc = inference(model, dl, criterion, valid_flg)
        if i_fold == 0:
            avg_pred_logits = logits
        else:
            avg_pred_logits += logits
    avg_pred_logits /= config['n_fold']
    all_preds, all_preds_prob = preds_class_prob(avg_pred_logits, dl)
    df_pred = post_process_pred(df, all_preds, all_preds_prob)
    return df_pred

def post_process_pred(df, all_preds, all_preds_prob):
    final_preds = []
    for i in range(len(df)):
        idx = df.id.values[i]
        pred = all_preds[i]
        pred_prob = all_preds_prob[i]
        j = 0
        while j < len(pred):
            cls = pred[j]
            if cls == '0': j += 1
            else: cls = cls.replace('B', 'I')
            end = j + 1
            while end < len(pred) and pred[end] == cls:
                end += 1
            if cls != '0' and cls !='':
                avg_score = np.mean(pred_prob[j:end])
                if end - j > config['MIN_THRESH'][cls] and avg_score > config['PROB_THRESH'][cls]:
                    final_preds.append((idx, cls.replace('I-', ''), ' '.join(map(str, list(range(j, end))))))
            j = end
    df_pred = pd.DataFrame(final_preds)
    df_pred.columns = ['id', 'class', 'new_predictionstring']
    return df_pred

def train_fn(model, dl_train, optimizer, epoch, criterion):
    lrs = []
    model.train()
    train_loss = 0
    train_f1sc = 0
    stream = tqdm(dl_train)
    scaler = GradScaler()

    for batch_idx, batch in enumerate(stream, start = 1):
        ids = batch['input_ids'].to(device, dtype = torch.long)
        mask = batch['attention_mask'].to(device, dtype = torch.long)
        raw_labels = batch['labels'].to(device, dtype = torch.long)
        word_ids = batch['word_ids'].to(device, dtype = torch.long)
        optimizer.zero_grad()
        with autocast():
            raw_logits = model(input_ids = ids, mask = mask)
        
        logits = active_logits(raw_logits, word_ids)
        labels = active_labels(raw_labels)
        preds, preds_prob = active_preds_prob(logits)
        train_f1sc += f1_score(labels.cpu().numpy(), preds.cpu().numpy(), average='macro')
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        lr=optimizer.param_groups[0]["lr"]
        lrs.append(lr) 
        scaler.update()
        train_loss += loss.item()
        
        if batch_idx % config['verbose_steps'] == 0:
            loss_step = train_loss / batch_idx
            print(f'Training loss after {batch_idx:04d} training steps: {loss_step}')
            
    epoch_loss = train_loss / batch_idx
    epoch_f1sc = train_f1sc / batch_idx
    del dl_train, raw_logits, logits, raw_labels, preds, labels
    torch.cuda.empty_cache()
    gc.collect()
    print(f'epoch {epoch} - training loss: {epoch_loss:.4f}')
    print(f'epoch {epoch} - training f1score: {epoch_f1sc:.4f}')
    return lrs


def valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion):
    oof, valid_loss, valid_f1  = get_preds_onefold(model, df_val, dl_val, criterion, valid_flg=True)
    f1score =[]
    # classes = oof['class'].unique()
    classes = ['Lead', 'Position', 'Counterclaim', 'Rebuttal','Evidence','Concluding Statement']
    print(f"Validation F1 scores")

    for c in classes:
        pred_df = oof.loc[oof['class'] == c].copy()
        gt_df = df_val_eval.loc[df_val_eval['discourse_type'] == c].copy()
        f1 = score_feedback_comp(pred_df, gt_df)
        print(f' * {c:<10}: {f1:4f}')
        f1score.append(f1)
    f1avg = np.mean(f1score)
    print(f'Overall Validation avg F1: {f1avg:.4f} val_loss:{valid_loss:.4f} val_f1:{valid_f1:.4f}')
    return valid_loss, oof




if __name__ == "__main__":

    # # Parse arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument('-n', '--name', required=True,
    #                     help='Name of the model to run. Available are: ' + \
    #                     'longformer, spacy_ner')


    # args = parser.parse_args()

    # Logging
    logger = logging.getLogger(f'TrainModel')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(colored('[%(asctime)s]', 'magenta') +
                                  colored('[%(levelname)s] ','blue') + 
                                  '%(message)s', '%Y-%m-%d %H:%M:%S')
 
    logging_file_handler = logging.FileHandler(
        os.path.join('./',f'{datetime.datetime.now().strftime("%Y%m%d_%H%M%S")}.log'))
    logging_file_handler.setLevel(logging.DEBUG)
    logging_file_handler.setFormatter(formatter)
    logger.addHandler(logging_file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    logger.info(f'Starting model training PID: ' + colored(f'{os.getpid()}', 'green'))
    train_start_time = time.time()


    df_alltrain = pd.read_csv(f'{config["data_dir"]}/corrected_train.csv')

    alltrain_texts = preprocess(df_alltrain)
    test_texts = preprocess()

    alltrain_texts = split_fold(alltrain_texts)
    alltrain_texts.head()


    seed_everything()
    # device optimization
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    early_stopping = EarlyStopping()

    print(f'Using device: {device}')
    
    oof = pd.DataFrame()
    for i_fold in range(config['n_fold']):
        print(f'=== fold{i_fold} training ===')
        tokenizer = LongformerTokenizerFast.from_pretrained(config['model_name'], add_prefix_space = True)
        model = FeedbackModel()
        model = model.to(device)
        
        
        df_train = alltrain_texts[alltrain_texts['fold'] != i_fold].reset_index(drop = True)
        ds_train = FeedbackPrizeDataset(df_train, tokenizer, config['max_length'], True)
        df_val = alltrain_texts[alltrain_texts['fold'] == i_fold].reset_index(drop = True)
        val_idlist = df_val['id'].unique().tolist()
        df_val_eval = df_alltrain.query('id==@val_idlist').reset_index(drop=True)
        ds_val = FeedbackPrizeDataset(df_val, tokenizer, config['max_length'], True)
        dl_train = DataLoader(ds_train, batch_size=config['train_batch_size'], shuffle=True, num_workers=2, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=config['valid_batch_size'], shuffle=False, num_workers=2, pin_memory=True)

        best_val_loss = np.inf
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(params=model.parameters(), lr=0.01)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.1)
        
        for epoch in range(1, config['n_epoch'] + 1):
            lrs = train_fn(model, dl_train, optimizer, epoch, criterion)
            valid_loss, _oof = valid_fn(model, df_val, df_val_eval, dl_val, epoch, criterion)
            scheduler.step()
            early_stopping(valid_loss)
            if early_stopping.early_stop:
                break
            if valid_loss < best_val_loss:
                best_val_loss = valid_loss
                _oof_fold_best = _oof
                _oof_fold_best['fold'] = i_fold
                model_filename = f'{config["model_dir"]}/{config["model_savename"]}_{i_fold}.bin'
                torch.save(model.state_dict(), model_filename)
                print(f'{model_filename} saved')
        
        
        
        plt.plot(range(1, config['n_epoch'] + 1),lrs)
        plt.title("PyTorch Learning Rate")
        plt.xlabel("epoch")
        plt.ylabel("learning rate")
        plt.savefig(f'{config["model_savename"]}.jpg')

        oof = pd.concat([oof, _oof_fold_best])


    oof.head()

    oof.to_csv(f'{config["output_dir"]}/oof_{config["name"]}.csv', index=False)

    pd.read_csv(f'{config["output_dir"]}/oof_{config["name"]}.csv').head()

    if config['is_debug']:
        idlist = alltrain_texts['id'].unique().tolist()
        df_train = df_alltrain.query('id==@idlist')
    else:
        df_train = df_alltrain.copy()
    print(f'overall cv score: {oof_score(df_train, oof)}')