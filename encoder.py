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


class DeepTriageNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, n_layers, bidirectional, dropout, device):     
        super(DeepTriageNN, self).__init__()
        self.device = device
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        #encoding the deep representation of the input stack trace
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size*2, output_size)
        self.relu = nn.ReLU()

    def forward(self, stacktrace, stacktrace_lengths):
        #stacktrace = (batch_size, length_stack)
        embedded = self.embedding(stacktrace)
        #embedded = (batch_size, stack_length,embedding_dim)
        packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, stacktrace_lengths, batch_first=True)
        packed_output, (hidden_state, cell_state) = self.lstm(packed_embedded)
        hidden = torch.cat((hidden_state[-2,:,:], hidden_state[-1,:,:]), dim = 1)
        dense_outputs=self.fc(hidden)
        dense_outputs = self.relu(dense_outputs)
        #outputs=self.sigmoid(dense_outputs)
        return dense_outputs



