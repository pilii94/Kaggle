import torch
import sys

sys.path.insert(0,'../src/')
from config import config

def active_logits(raw_logits, word_ids):
    word_ids = word_ids.view(-1)
    active_mask = word_ids.unsqueeze(1).expand(word_ids.shape[0], config['num_labels'])
    active_mask = active_mask != config['NON_LABEL']
    active_logits = raw_logits.view(-1, config['num_labels'])
    active_logits = torch.masked_select(active_logits, active_mask) # return 1dTensor
    active_logits = active_logits.view(-1, config['num_labels']) 
    return active_logits

def active_labels(labels):
    active_mask = labels.view(-1) != config['IGNORE_INDEX']
    active_labels = torch.masked_select(labels.view(-1), active_mask)
    return active_labels

def active_preds_prob(active_logits):
    active_preds = torch.argmax(active_logits, axis = 1)
    active_preds_prob, _ = torch.max(active_logits, axis = 1)
    return active_preds, active_preds_prob