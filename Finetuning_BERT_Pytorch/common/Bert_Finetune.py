import spacy
import glob
import os
import yaml
from torch.utils.data import DataLoader, Dataset
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm
from tqdm import tqdm
from pathlib import Path
import pandas as pd
import logging
import random
import numpy as np
import pickle
from termcolor import colored
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import tensorflow as tf
from sklearn.model_selection import train_test_split
from transformers import BertTokenizerFast, BertForSequenceClassification, AdamW, TFBertModel
from transformers import get_linear_schedule_with_warmup
from sklearn.metrics import mean_squared_error
from train_config import config

import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()                             

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()                

if torch.cuda.is_available():        
    device = torch.device("cuda")
    print('There are %d GPU(s) available.' % torch.cuda.device_count())
    print('We will use the GPU:', torch.cuda.get_device_name(0))

else:
    print('No GPU available, please check.')
    # device = torch.device("cpu")

class BertClassifier():
    def __init__(self, logger=None):
        self.logger = logger
        self.seed_val = 42
        self.model = None
        self.is_trained = False
        self.model_name = config['model_name']
        self.tokenizer = BertTokenizerFast.from_pretrained(self.model_name, do_lower_case=True)
        self.MAX_LEN = config['MAX_LEN'] 
        self.lr = config['lr'] 
        self.eps = config['eps'] 
        self.batch_size = config['batch_size']  
        self.epochs = config['epochs']
        
        
    def process_sentences(self, X):
        #Sents to ids, padding and truncating
        input_ids = []
        for sent in X:
            encoded_sent = self.tokenizer.encode(sent, add_special_tokens = True)
            input_ids.append(encoded_sent)
        input_ids = tf.keras.preprocessing.sequence.pad_sequences(input_ids, maxlen=self.MAX_LEN, dtype="long", 
                            value=0, truncating="pre", padding="pre")#pre, post
        return input_ids
        
    def create_attentionmasks(self, input_ids):
        # Create attention masks
        attention_masks = []
        for sent in input_ids:
            att_mask = [int(token_id > 0) for token_id in sent]
            attention_masks.append(att_mask)
        return attention_masks

    def print_rmse(self, preds, labels):
#         pred_flat = np.argmax(preds, axis=1).flatten()
#         labels_flat = labels.flatten()

        return  np.sqrt(mean_squared_error(labels, preds))
    


        
    def fit(self, X,y):
        """
        This function trains a model using pretrained Bert
        """
        input_ids = self.process_sentences(X)
  
        attention_masks = self.create_attentionmasks(input_ids)
        # Use 90% for training and 10% for validation.
        train_inputs, validation_inputs, train_labels, validation_labels = train_test_split(input_ids, y, 
                                                                    random_state=2018, test_size=0.1)
        # Do the same for the masks.
        train_masks, validation_masks, _, _ = train_test_split(attention_masks, y,
                                                    random_state=2018, test_size=0.1)
        # Convert into torch 
        train_inputs = torch.tensor(train_inputs)
        validation_inputs = torch.tensor(validation_inputs)
        
        train_masks = torch.tensor(train_masks)
        validation_masks = torch.tensor(validation_masks)
       
        train_labels = torch.tensor(train_labels.to_numpy(),dtype=torch.float)
        validation_labels = torch.tensor(validation_labels.to_numpy(),dtype=torch.float)

         # Create the DataLoader tr.
        train_data = TensorDataset(train_inputs, train_masks, train_labels)
        train_sampler = RandomSampler(train_data)
        train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=self.batch_size)
        # Create the DataLoader val.
        validation_data = TensorDataset(validation_inputs, validation_masks, validation_labels)
        validation_sampler = SequentialSampler(validation_data)
        validation_dataloader = DataLoader(validation_data, sampler=validation_sampler, batch_size=self.batch_size)



        basemodel = BertForSequenceClassification.from_pretrained("bert-base-uncased",
                                                num_labels = 1, 
                                                output_attentions = False, 
                                                output_hidden_states = False)
        
        basemodel.cuda()
        # optimizer = torch.optim.Adam(basemodel.parameters(), 
        #           lr = self.lr, 
        #           eps = self.eps 
        #         ) 

        # optimizer = torch.optim.SGD(basemodel.parameters(),
        #     lr = self.lr
        # ) 

        optimizer = AdamW(basemodel.parameters(), # Implements Adam algorithm with weight decay fix as introduced in Decoupled Weight Decay Regularization.
                  lr = self.lr, 
                  eps = self.eps 
                ) 

        total_steps = len(train_dataloader) * self.epochs

        #Create a schedule with a learning rate that decreases linearly from the initial lr set in the optimizer to 0, after a warmup period during which it increases linearly from 0 to the initial lr set in the optimizer
        scheduler = get_linear_schedule_with_warmup(optimizer, 
                                            num_warmup_steps = 0, 
                                            num_training_steps = total_steps)
        # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1, verbose=True)
        

        random.seed(self.seed_val)
        np.random.seed(self.seed_val)
        torch.manual_seed(self.seed_val)
        torch.cuda.manual_seed_all(self.seed_val)
        # Store the average loss
        loss_values = []

        es = EarlyStopping(patience=10, logger = self.logger)

        for epoch_i in range(0, self.epochs):
            
            self.logger.info(f'============= Epoch: {epoch_i + 1} / {self.epochs} =============')
            self.logger.info('Training')
            total_loss = 0
            basemodel.train()
            for step, batch in enumerate(train_dataloader):
                b_input_ids = batch[0].to(device)
                b_input_mask = batch[1].to(device)
                b_target = batch[2].to(device)
                basemodel.zero_grad() 

                outputs = basemodel(b_input_ids, 
                    token_type_ids=None, 
                    attention_mask=b_input_mask, 
                    labels=b_target)
                loss = outputs[0]
                total_loss += loss.item()
                loss.backward() #calc gradients
                torch.nn.utils.clip_grad_norm_(basemodel.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
            avg_train_loss = total_loss / len(train_dataloader) 
            loss_values.append(avg_train_loss)
            
            self.logger.info(f"Avg training loss: {avg_train_loss}")
            self.logger.info("Running Validation")
            basemodel.eval()
            eval_rmse = 0
            nb_eval_steps, nb_eval_examples = 0, 0
            for batch in validation_dataloader:
                batch = tuple(t.to(device) for t in batch)
                b_input_ids, b_input_mask, b_target = batch
                with torch.no_grad():
                    outputs = basemodel(b_input_ids, 
                            token_type_ids=None, 
                            attention_mask=b_input_mask)
                logits = outputs[0]
                logits = logits.detach().cpu().numpy()
                label_ids = b_target.to('cpu').numpy()

                tmp_eval_rmse= self.print_rmse(logits, label_ids)
                eval_rmse += tmp_eval_rmse

                nb_eval_steps += 1
            # Report the final accuracy for this validation run.
            self.logger.info(f"RMSE: {eval_rmse/nb_eval_steps}")
          
            
            if es.step(eval_rmse, basemodel, self.tokenizer, epoch_i):
                self.logger.info("Early stopping.")
                break  # early stop criterion is met, we can stop now 
        
        self.logger.info("Training complete.")
        self.logger.info("Retrieving best model.")
        self.model, self.tokenizer = es.get_best_model()
        self.is_trained = True



      
    def predict(self, X):
        assert self.is_trained, 'Model should be trained before inference.'
        input_ids = self.process_sentences(X)
        attention_masks = self.create_attentionmasks(input_ids)
        prediction_inputs = torch.tensor(input_ids)
        prediction_masks = torch.tensor(attention_masks)
       
        # DataLoader.
        prediction_data = TensorDataset(prediction_inputs, prediction_masks)
        prediction_sampler = SequentialSampler(prediction_data)
        prediction_dataloader = DataLoader(prediction_data, sampler=prediction_sampler, batch_size=self.batch_size)

        # evaluation mode
        self.model.eval()
        preds = []
        predictions = []
        final_preds = []#np.array([]) 
        # Predict 
        for batch in prediction_dataloader:
            
            batch = tuple(t.to(device) for t in batch)
            b_input_ids, b_input_mask = batch
            
            with torch.no_grad():
                output = self.model(b_input_ids, token_type_ids=None, 
                                attention_mask=b_input_mask)
#             logits = outputs[0]
          
#             logits = logits.detach().cpu().numpy()
#             predictions.append(logits)
                output = output["logits"].squeeze(-1)
                preds.append(output.cpu().numpy())

        predictions = np.concatenate(preds)
        
        return predictions
#         for i in range(len(predictions)):
#             pred_labels_i = np.argmax(predictions[i], axis=1).flatten()
#             final_preds = np.concatenate((final_preds, pred_labels_i), axis=0)
        
#         return  np.asarray(final_preds)
        
       
    def save(self, path):
        if self.is_trained:
            output_dir = Path(path)
            if not output_dir.exists():
                os.makedirs(output_dir)
            model_config = {}
            # serialize model 
            with open(os.path.join(output_dir, f'model_config.yaml'), 'w') as file:
                documents = yaml.dump(model_config, file)

            model_to_save = self.model.module if hasattr(self.model, 'module') else self.model  
            model_to_save.save_pretrained(str(output_dir))
            self.tokenizer.save_pretrained(str(output_dir))

            self.logger.info(f'Saved model to {output_dir}')
        else:
            self.logger.warning('Cannot save the model. Train it first.')

    def load(self, path):
        output_dir = Path(path)
        with open(os.path.join(output_dir, f'model_config.yaml')) as file:
            model_config = yaml.load(file, Loader=yaml.FullLoader)
        
        # Load trained model and vocabulary fine-tuned
        self.model = BertForSequenceClassification.from_pretrained(str(output_dir),num_labels=1)
        self.tokenizer = BertTokenizerFast.from_pretrained(str(output_dir))

        # Copy the model to the GPU. Check if prediction is ok in CPU
        self.model.to(device)
       
        self.is_trained = True


class EarlyStopping(object):
    def __init__(self, logger = None, mode='min', min_delta=0, patience=10, percentage=False):
        self.mode = 'min'
        self.logger = logger
        self.best_model = None
        self.best_tokenizer = None
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta, percentage)

        if patience == 0:
            self.is_better = lambda a, b: True
            self.step = lambda a: False

    def step(self, metrics, basemodel, tokenizer, epoch):
        if epoch == 0:
            self.best_model = basemodel
            self.best_tokenizer = tokenizer

        if self.best is None:
            self.best = metrics
            return False

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
            self.logger.info(f'Creating model checkpoint')
            # self.model_checkpoint(basemodel, tokenizer)
            self.best_model = basemodel
            self.best_tokenizer = tokenizer

        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode='min', min_delta=0, percentage=False):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if not percentage:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - min_delta
            if mode == 'max':
                self.is_better = lambda a, best: a > best + min_delta
        else:
            if mode == 'min':
                self.is_better = lambda a, best: a < best - (
                            best * min_delta / 100)
            if mode == 'max':
                self.is_better = lambda a, best: a > best + (
                            best * min_delta / 100)
    def get_best_model(self):
        return self.best_model, self.best_tokenizer
    
if __name__ == '__main__':
    # Logging
    logger = logging.getLogger('BertClassifier')
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter(colored('[%(asctime)s]', 'magenta') + colored('[%(levelname)s] ',
                                                                                'blue') + '%(message)s',
                                  '%Y-%m-%d %H:%M:%S')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler) 
    
    
    data_train = pd.read_csv(config['train_data'])
    data_train.reset_index(drop=True)

    out_path = os.path.join(config['output_dir'], config['model_name'],'v1')
    model = BertClassifier(logger)
    model.fit(data_train["excerpt"],data_train["target"])
    model.save(out_path)  

    print("Finished training. Model saved.")
