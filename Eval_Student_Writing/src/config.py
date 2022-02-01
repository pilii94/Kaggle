import os

config = {}

config['name'] = 'fp_exp3'
config['model_savename'] = 'longformer'

config['model_name'] = 'allenai/longformer-base-4096' # download from Internet
config['base_dir'] = '../'
config['data_dir'] = os.path.join(config['base_dir'], 'data/')
config['pre_data_dir'] = os.path.join(config['base_dir'], 'data/preprocessed')
config['model_dir'] = os.path.join(config['base_dir'], f'model/{config["name"]}')
config['output_dir'] = os.path.join(config['base_dir'], f'output/{config["name"]}')
config['is_debug'] = False
config['n_epoch'] = 1 # not to exceed runtime limits on Kaggle
config['n_fold'] = 5
config['verbose_steps'] = 500
config['random_seed'] = 42
config['max_length'] = 1024
config['train_batch_size'] = 4
config['valid_batch_size'] = 4
config['lr'] = 5e-5
config['num_labels'] = 15
config['label_subtokens'] = True
config['output_hidden_states'] = True
config['hidden_dropout_prob'] = 0.1
config['layer_norm_eps'] = 1e-7
config['add_pooling_layer'] = False
config['verbose_steps'] = 500
if config['is_debug']:
    config['debug_sample'] = 1000
    config['verbose_steps'] = 16
    config['n_epoch'] = 1
    config['n_fold'] = 2


config['IGNORE_INDEX'] = -100
config['NON_LABEL'] = -1
config['OUTPUT_LABELS'] = ['0', 'B-Lead', 'I-Lead', 'B-Position', 'I-Position', 'B-Claim', 'I-Claim', 'B-Counterclaim', 'I-Counterclaim', 
                 'B-Rebuttal', 'I-Rebuttal', 'B-Evidence', 'I-Evidence', 'B-Concluding Statement', 'I-Concluding Statement']
config['LABELS_TO_IDS'] = {v:k for k,v in enumerate(config['OUTPUT_LABELS'])}
config['IDS_TO_LABELS'] = {k:v for k,v in enumerate(config['OUTPUT_LABELS'])}

config['MIN_THRESH'] = {
    "I-Lead": 9,
    "I-Position": 5,
    "I-Evidence": 14,
    "I-Claim": 3,
    "I-Concluding Statement": 11,
    "I-Counterclaim": 6,
    "I-Rebuttal": 4,
}

config['PROB_THRESH'] = {
    "I-Lead": 0.7,
    "I-Position": 0.55,
    "I-Evidence": 0.65,
    "I-Claim": 0.55,
    "I-Concluding Statement": 0.7,
    "I-Counterclaim": 0.5,
    "I-Rebuttal": 0.55,
}
