import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import torch
from torch import nn, optim
import time
from funcs import *
from net import *


def detect_torch(net, params, train,validation=[], test=[], val_exist=False, test_exist=False, preprocess= preprocess.PK_log, batchsize=32, learning_rate=0.001, momentum=0.9,step_size=10, gamma=0.5, num_epochs=10, save_path="", use_gpu=False, pretrained=True):
    criterion = Logloss()
    
#     nn.MSELoss()
#     Logloss()
#     
#     nn.KLDivLoss()

#     use_gpu = torch.cuda.is_available()
   
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=step_size,gamma=gamma)
    start_epoch=0
#     lambda1 = lambda epoch: 0.8 ** epoch
#     scheduler = optim.lr_scheduler.LambdaLR(optimizer,lr_lambda=lambda1)


# Load Checkpoint:
    if pretrained:
        checkpoint = torch.load(save_path+'model.pth')
        net= checkpoint['model']
        net.load_state_dict(checkpoint['state_dict'])
        start_epoch = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
    
    train_loss=[]
    train_time=[]
        
    val_loss=[]
    val_time=[]
    l0=100  #sample initial epoch loss

    for epoch in range(num_epochs)[start_epoch:]:
        total_loss=0
        start_time=time.time()


        net.train()
#         scheduler.step()
        print(optimizer.param_groups[0]["lr"])

        batches=random_mini_batches(train['Density_Field'],train[params],batchsize)
    
        for batch in batches:
            datafile, labels = batch
            inputs, labels = preprocess(datafile, labels)
#             labels = torch.tensor(labels.values)*100
            
            if use_gpu:
                inputs, labels = inputs.cuda(),labels.cuda() 
#             print(inputs.shape)

#                 print(labels.shape)
            optimizer.zero_grad()

            outputs = net(inputs)
#             print(outputs.shape)
            loss = criterion(outputs.type(torch.float64), labels)
            loss.backward()
            optimizer.step()
        
            total_loss += loss.item() 

            
            print("Train Batch Loss: {} " .format(loss.item()))
            
        stop_time=time.time() 
        epoch_loss=total_loss/len(batches)
        epoch_time=stop_time-start_time
        train_loss.append(epoch_loss)
        train_time.append(epoch_time)

        print("Epoch {} :     Train Loss = {}       Train Time {}  " 
                .format(epoch+1,epoch_loss,epoch_time))
        
        
    
        if val_exist==True:
            with torch.no_grad():
                total_loss=0
                start_time=time.time()
                net.eval()

                
                batches=random_mini_batches(validation['Density_Field'],validation[params],batchsize)
                for batch in batches:
                    datafile, labels = batch
                    inputs, labels = preprocess(datafile, labels)
#                     labels = torch.tensor(labels.values)*100
                    
                    if use_gpu:
                        inputs, labels = inputs.cuda(),labels.cuda() 
                        
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    
                    total_loss += loss.item() 
                    

                    print("Validation Batch Loss: {} " .format(loss.item()))
                    
                stop_time=time.time()        
                epoch_loss=total_loss/len(batches)
                epoch_time=stop_time-start_time
                val_loss.append(epoch_loss)
                val_time.append(epoch_time)
            
                print("Epoch {} :    Validation Loss = {}   Validation Time = {}" 
                  .format(epoch+1,epoch_loss,epoch_time))
        

        
        
        if epoch_loss<l0:
            l0=epoch_loss
            checkpoint = {'model': net,
                          'epoch': epoch+1,
                          'state_dict': net.state_dict(),
                          'optimizer' : optimizer.state_dict()}

            torch.save(checkpoint, save_path+'model.pth')
#             torch.save(resnet.state_dict(),image_path)
            print("Model saved in file: %s" % save_path)
            
        
#             wandb.log({
#                 'epoch1': epoch, 
#                 'train_loss1': train_loss[-1]*1e5, 
#                 'val_loss1': val_loss[-1]*1e5
#                   })
    
        scheduler.step()
        
    if pretrained:
        df=pd.DataFrame.from_dict({'train_loss':train_loss, 'train_time':train_time})
        df_train=pd.read_csv(save_path+'train_result.csv')
        df_train=pd.concat([df_train,df], axis=0).reset_index()
        df_train.to_csv(save_path+'train_result.csv')
    else:    
        df_train=pd.DataFrame.from_dict({'train_loss':train_loss, 'train_time':train_time})
        df_train.to_csv(save_path+'train_result.csv')
        
    if val_exist==True:
        if pretrained:
            df=pd.DataFrame.from_dict({'val_loss':val_loss, 'val_time':val_time})
            df_val=pd.read_csv(save_path+'val_result.csv')
            df_val=pd.concat([df_val,df], axis=0).reset_index()
            df_val.to_csv(save_path+'val_result.csv')
        else:
            df_val=pd.DataFrame.from_dict({'val_loss':val_loss, 'val_time':val_time})
            df_val.to_csv(save_path+'val_result.csv')
            
    if test_exist==True:
        with torch.no_grad():
            total_loss=0
            start_time=time.time()
            net.eval()
            
            
            batches=random_mini_batches(test['Density_Field'],test[params],batchsize)
            for batch in batches:     
                datafile, labels = batch
                inputs, labels = preprocess(datafile, labels)
#                 labels = torch.tensor(labels.values)*100
                
                if use_gpu:
                    inputs, labels = inputs.cuda(),labels.cuda() 
                        
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                    
                total_loss += loss.item() 
                
                print("Test Batch Loss: {} " .format(loss.item()))
                
            stop_time=time.time()        
            test_loss=total_loss/len(batches)
            test_time=stop_time-start_time
    
            print("Test Loss = {}       Test Time = {}" 
                  .format(test_loss,test_time)) 
            
            df_test=pd.DataFrame.from_dict({'test_loss':[test_loss], 'test_time':[test_time]})
            df_test.to_csv(save_path+'test_result.csv')



            