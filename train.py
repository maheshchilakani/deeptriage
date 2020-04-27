def train(model, iterator, optimizer, criterion):
    
    #initialize every epoch 
    epoch_loss = 0
    epoch_acc = 0
    
    #set the model in training phase
    model.train()  
    
    for batch in iterator:
        #resets the gradients after every batch
        optimizer.zero_grad()   
        #retrieve text and no. of words
        text, text_lengths = batch.stacktrace   
        #convert to 1D tensor
        predictions = model(text, text_lengths).squeeze()  
        #compute the loss
        print(predictions)
        print("***Batch")
        print(batch.icmteam)
        loss = criterion(predictions, batch.icmteam)        
        #compute the binary accuracy
        acc = multi_acc(predictions, batch.icmteam)   
        #backpropage the loss and compute the gradients
        loss.backward()       
        
        #update the weights
        optimizer.step()      
        
        #loss and accuracy
        epoch_loss += loss.item()  
        epoch_acc += acc.item()    
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def multi_acc(y_pred, y_test):
    y_pred_softmax = torch.log_softmax(y_pred, dim = 1)
    _, y_pred_tags = torch.max(y_pred_softmax, dim = 1)    
    correct_pred = (y_pred_tags == y_test).float()
    acc = correct_pred.sum() / len(correct_pred)
    acc = torch.round(acc) * 100
    return acc
    