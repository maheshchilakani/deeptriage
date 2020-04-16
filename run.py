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

import numpy as np 
import math 
import random 
import re 
import os 
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import spacy
import numpy as np

import random
import math
import time

def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]

SEED = 1234

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True

spacy_en = spacy.load('en')


STACKTRACE = data.Field(tokenize=tokenize_en,batch_first=True,include_lengths=True, lower= True)
ICMTEAM = data.LabelField(dtype = torch.float,batch_first=True)

dataset_fields = [('text', ICMTEAM ), ('test', STACKTRACE)]

training_data = data.TabularDataset(path = '.\data\SampleTriageData.csv', format = 'csv', fields = dataset_fields, skip_header= False)

print(vars(training_data.examples[0]))


train_data, valid_data = training_data.split(split_ratio=0.8, random_state=random.seed(SEED))

#Building Vocabularies for boththe icmTeams and the Stacktrace for the team. 
STACKTRACE.build_vocab(train_data, min_freq=2, vectors= "glove.6B.100d")
ICMTEAM.build_vocab(train_data)
STACKTRACE.vocab
print(len(STACKTRACE.vocab))
