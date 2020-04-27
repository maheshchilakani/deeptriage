import torch 
import torch.nn as nn 

loss = nn.CrossEntropyLoss()

Y = torch.tensor([0])
Y_pred = torch.tensor([[2.0, 1.0, 0.1]])

print(loss(Y_pred, Y))