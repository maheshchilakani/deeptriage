#torch base function package
import torch 
#torch nn class definition. We will inherit Module class from this
#this will help do the leg work of tracking and updating grads. Also 
#helpps in setting
import torch.nn as nn 
#utility class/method 
from torchtext import data
from torch.utils.data import DataLoader, Dataset 
#optimizers to run the grad descent optimally based on the chosen algo.  
import  torch.optim as optim
from torchtext.data import Field, BucketIterator

import math 
import random 
import re 
import os 
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy

import random
import math
import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from encoder import DeepTriageNN
from vocabulary import Vocabulary 
from train import *
from evaluate import *

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

##Hyperparameter
SEED = 1234
EMBEDDING_DIM = 100
BATCH_SIZE = 1
HIDDEN_SIZE = 32
N_LAYERS = 2
DROP_OUT = 0.3
EPOCHS = 10 

random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en')


STACKTRACE = data.Field(tokenize=tokenize_en, batch_first=True, include_lengths=True, lower= True)
ICMTEAM = data.LabelField(dtype=torch.float, batch_first=True)

dataset_fields = [('icmteam', ICMTEAM ), ('stacktrace', STACKTRACE)]

training_data = data.TabularDataset(path = '.\data\SampleTriageData.csv', format = 'csv', fields = dataset_fields, skip_header= False)

#print(vars(training_data.examples[4]))

train_data, valid_data = training_data.split(split_ratio=0.8, random_state=random.seed(SEED))

#Building Vocabularies for boththe icmTeams and the Stacktrace for the team. 
STACKTRACE.build_vocab(train_data, min_freq=2, vectors= "glove.6B.100d")
ICMTEAM.build_vocab(train_data)
#print(ICMTEAM.vocab.stoi)


#check whether cuda is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

#set batch size
BATCH_SIZE = 4

#Load an iterator
train_iterator, valid_iterator = data.BucketIterator.splits(
    (train_data, valid_data), 
    batch_size = BATCH_SIZE,
    sort_key = lambda x: len(x.stacktrace),
    sort_within_batch=True,
    device = device)

model = DeepTriageNN(len(STACKTRACE.vocab), embedding_dim = EMBEDDING_DIM, hidden_size = HIDDEN_SIZE, output_size = len(ICMTEAM.vocab),
                     n_layers = N_LAYERS, bidirectional = True, dropout = DROP_OUT, device = device)

print(model)

#No. of trianable parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
print(f'The model has {count_parameters(model):,} trainable parameters')

#Initialize the pretrained embedding
pretrained_embeddings = STACKTRACE.vocab.vectors
model.embedding.weight.data.copy_(pretrained_embeddings)

accuracy_stats = {
    'train': [],
    "val": []
}
loss_stats = {
    'train': [],
    "val": []
}


#define optimizer and loss
optimizer = optim.Adam(model.parameters())
criterion = nn.CrossEntropyLoss()

#define metric
def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc
    
#push to cuda if available
model = model.to(device)
criterion = criterion.to(device)


best_valid_loss = float('inf')

for epoch in range(EPOCHS):
     
    #train the model
    train_loss, train_acc = train(model, train_iterator, optimizer, criterion)
    loss_stats['train'].append(train_loss)
    accuracy_stats['train'].append(train_acc)
    #evaluate the model
    valid_loss, valid_acc = evaluate(model, valid_iterator, criterion)
    loss_stats['val'].append(valid_loss)
    accuracy_stats['val'].append(valid_acc)
    print(f'Epoch {(epoch+1)+0:03}: | Train Loss: {train_loss:.5f} | Val Loss: {valid_loss:.5f} | Train Acc: {train_acc:.3f}| Val Acc: {valid_acc:.3f}')

    #save the best model
    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), 'saved_weights.pt')
    
    

# Create dataframes
train_val_acc_df = pd.DataFrame.from_dict(accuracy_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
train_val_loss_df = pd.DataFrame.from_dict(loss_stats).reset_index().melt(id_vars=['index']).rename(columns={"index":"epochs"})
# Plot the dataframes
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(20,7))
sns.lineplot(data=train_val_acc_df, x = "epochs", y="value", hue="variable",  ax=axes[0]).set_title('Train-Val Accuracy/Epoch')
sns.lineplot(data=train_val_loss_df, x = "epochs", y="value", hue="variable", ax=axes[1]).set_title('Train-Val Loss/Epoch')