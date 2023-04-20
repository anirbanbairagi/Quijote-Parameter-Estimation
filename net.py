import torch
from torch import nn


class Logloss(nn.Module):
    def __init__(self):
        super(Logloss, self).__init__()

    def forward(self, output, target):
        criterion = nn.MSELoss()
        total_log_loss = 0
        dim = output.shape[1]
        for i in range(dim):
            coloumn_loss = criterion(output[:,i], target[:,i])
            total_log_loss += torch.log(coloumn_loss)
         
        return total_log_loss


class ActiveBen(nn.Module):
   
    def __init__(self):
       
        super(ActiveBen, self).__init__() 

    def forward(self, input):
        '''
        Forward pass of the function.
        '''
        output = torch.zeros_like(input)
        idx1 = input<=-1
        output[idx1]= input[idx1]+1/3
        idx2 = (input<1)*(input>-1)
        output[idx2]= input[idx2]*(input[idx2]**2+3)/6
        idx3 = input>=1
        output[idx3]=input[idx3]-1/3
        
        return output

class Models():
    def Anirban():
        model=nn.Sequential(
            nn.Conv2d(128,256,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool2d(4,4),
        
            nn.Conv2d(256,512,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool2d(4,4),
        
            nn.Conv2d(512,1024,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool2d(4,4),
        
            nn.Flatten(),
            nn.Linear(1024,512),nn.Sigmoid(),
            nn.Linear(512,len(params))
        )
    
        return model
    
    def Anirban1():
        model=nn.Sequential(
            nn.Conv2d(128,64,kernel_size=4),nn.ReLU(),
            nn.MaxPool2d(4,4),
        
            nn.Conv2d(64,32,kernel_size=4),nn.ReLU(),
            nn.MaxPool2d(4,4),
        
            nn.Conv2d(32,16,kernel_size=4),nn.ReLU(),
            nn.MaxPool2d(4,4),
        
            nn.Flatten(),
            nn.Linear(16,128),nn.ReLU(),
            nn.Linear(128,len(params))
        )
    
        return model
    
    def Anirban2():
        model=nn.Sequential(
            nn.Conv3d(1,128,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool3d(4),
        
            nn.Conv3d(128,256,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool3d(4),
        
            nn.Conv3d(256,512,kernel_size=4),nn.Sigmoid(),
            nn.MaxPool3d(4),
        
            nn.Flatten(),
#             nn.Linear(128,128),nn.Sigmoid(),
            nn.Linear(512,len(params))
        )
    
        return model
    
    def Anirban4(params):
        model=nn.Sequential(
            nn.Conv3d(1,32,kernel_size=7),
            nn.AvgPool3d(3,3),
            ActiveBen(),
        
         
            nn.Conv3d(32,64,kernel_size=5),
            nn.AvgPool3d(3,3),
#             nn.MaxPool3d(2,2), 
            ActiveBen(),
            
            
            nn.Conv3d(64,128,kernel_size=3),
            nn.AvgPool3d(3,3),
#             nn.MaxPool3d(2,2), 
            ActiveBen(),
            
            
            
            
            nn.Conv3d(128,256,kernel_size=1),
            ActiveBen(),
            
            
          
        
            nn.Flatten(),         
            nn.Linear(6912,256), ActiveBen(),
            nn.Linear(256,len(params))
        )
    
        return model
    
    def Anirban4a():                                       #4a giving same value for omb
        model=nn.Sequential(
            nn.Conv3d(1,64,kernel_size=4),
            nn.MaxPool3d(4,4),
        
            nn.Conv3d(64,128,kernel_size=4),
            nn.MaxPool3d(4,4),
        
            nn.Conv3d(128,256,kernel_size=4),
            nn.MaxPool3d(4,4),
        
            nn.Flatten(),
#             nn.Linear(256,256),nn.Tanh(),         #128,128 - 128,out
            nn.Linear(256,128),nn.Tanh(),
#             nn.Linear(128,64),
            nn.Linear(128,len(params))
        )
    
        return model
    
    def Anirban6():
        model=nn.Sequential(
            nn.Conv3d(1,16,kernel_size=4),
            nn.AvgPool3d(4,4),
            nn.Softplus(),
        
            nn.Conv3d(16,32,kernel_size=4),
            nn.AvgPool3d(4,4),
            nn.Softplus(),
        
            nn.Conv3d(32,64,kernel_size=4),
            nn.AvgPool3d(4,4),
            nn.Softplus(),
            
#             nn.Conv3d(64,128,kernel_size=4),
#             nn.AvgPool3d(2,2),
#             nn.Softplus(),
        
            nn.Flatten(),
#             nn.Linear(128*125,128),nn.Softplus(),
#             nn.Linear(128,64),
            nn.Linear(64,32),
            nn.Linear(32,len(params))
        )
    
        return model
    
    def Anirban5():
        model=nn.Sequential(
            nn.Conv3d(1,32,kernel_size=16),
            nn.BatchNorm3d(32),
            nn.AvgPool3d(2,2),
#             nn.MaxPool3d(4,4),
        
            nn.Conv3d(32,64,kernel_size=16),
            nn.BatchNorm3d(64),
            nn.AvgPool3d(2,2),
#             nn.MaxPool3d(4,4),
        
            nn.Conv3d(64,128,kernel_size=16),
            nn.BatchNorm3d(128),
#             nn.MaxPool3d(4,4),
            
#             nn.Conv3d(128,256,kernel_size=16),
#             nn.BatchNorm3d(256),
            
#             nn.Conv3d(256,512,kernel_size=16),
#             nn.BatchNorm3d(512),
            
#             nn.Conv3d(512,1024,kernel_size=16),
#             nn.BatchNorm3d(1024),
#             nn.AvgPool3d(3,3),
        
            nn.Flatten(),
#             nn.Linear(1024*11*11,1024),
#             nn.Linear(1024,512),
            nn.Linear(64,128),
            nn.Linear(128,len(params))
        )
    
        return model
    
    def Anirban_pk(params):
        model=nn.Sequential(
            nn.Linear(128,1024), 
            ActiveBen(),
        
         
#             nn.Linear(128,256),
#             ActiveBen(),
            
            
#             nn.Linear(256,128), 
#             ActiveBen(),
            
            nn.Linear(1024,len(params))
        )
    
        return model
    
#     128-128, 128-256, 256-256, 256-128, 128-5. -37
    def Anirban_pk1(params):
        
        model=nn.Sequential(
            nn.Linear(128, 128), 
            ActiveBen(),
            
            nn.Linear(128, 256), 
            ActiveBen(),
        
            nn.Linear(256, 256), 
            ActiveBen(),     
        
            nn.Linear(256, 128), 
            ActiveBen(),
            
            nn.Linear(128,len(params))
        )
    
        return model

    
#     def Anirban_pk1(params):
        
#         model=nn.Sequential(
#             nn.Linear(128, 128), 
#             ActiveBen(),
            
#             nn.Linear(128, 256), 
#             ActiveBen(),
        
#             nn.Linear(256, 512), 
#             ActiveBen(),
            
#             nn.Linear(512, 512), 
#             ActiveBen(),
            
#             nn.Linear(512, 256), 
#             ActiveBen(),
        
#             nn.Linear(256, 128), 
#             ActiveBen(),
            
#             nn.Linear(128,len(params))
#         )
    
#         return model

    
class ConvBlock(nn.Module):    
    
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm3d(out_chanels)
        
    def forward(self, x):
        return ActiveBen()(self.bn(self.conv(x)))
    
    

class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionBlock, self).__init__()
        self.branch1 = ConvBlock(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock(in_channels, red_5x5, kernel_size=1),
            ConvBlock(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, padding=1, stride=1),
            ConvBlock(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
    
    
    
class InceptionAux(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool3d(kernel_size=5, stride=3)
        self.conv = ConvBlock(in_channels, 128, kernel_size=1)
#         self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(1024, 1024)
        self.fc2 = nn.Linear(1024, num_classes)
    
    def forward(self, x):
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
#         print(x.shape)

        x = ActiveBen()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    
    
    
class InceptionV1(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionV1, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock(in_channels=1, out_chanels=64,kernel_size=7,stride=2,padding=3)
        self.conv2 = ConvBlock(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool3d(kernel_size=3, stride=1)
#         self.avgpool = nn.AvgPool3d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(8192, num_classes)
#         self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux(512, num_classes)
            self.aux2 = InceptionAux(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
#         print(x.shape)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)
#         print(x.shape)
        x = self.inception5b(x)
#         print(x.shape)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x

    
    
## Pk - 1D Inception module

class ConvBlock_1D(nn.Module):    
    
    def __init__(self, in_channels, out_chanels, **kwargs):
        super(ConvBlock_1D, self).__init__()
        self.conv = nn.Conv1d(in_channels, out_chanels, **kwargs)
        self.bn = nn.BatchNorm1d(out_chanels)
        
    def forward(self, x):
        return ActiveBen()(self.bn(self.conv(x)))
    
    

class InceptionBlock_1D(nn.Module):
    def __init__(self, in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_pool):
        super(InceptionBlock_1D, self).__init__()
        self.branch1 = ConvBlock_1D(in_channels, out_1x1, kernel_size=1)
        self.branch2 = nn.Sequential(
            ConvBlock_1D(in_channels, red_3x3, kernel_size=1, padding=0),
            ConvBlock_1D(red_3x3, out_3x3, kernel_size=3, padding=1),
        )
        self.branch3 = nn.Sequential(
            ConvBlock_1D(in_channels, red_5x5, kernel_size=1),
            ConvBlock_1D(red_5x5, out_5x5, kernel_size=5, padding=2),
        )
        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, padding=1, stride=1),
            ConvBlock_1D(in_channels, out_pool, kernel_size=1),
        )
    
    def forward(self, x):
        branches = (self.branch1, self.branch2, self.branch3, self.branch4)
        return torch.cat([branch(x) for branch in branches], 1)
    
    
    
class InceptionAux_1D(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(InceptionAux_1D, self).__init__()
        self.dropout = nn.Dropout(p=0.7)
        self.pool = nn.AvgPool1d(kernel_size=5, stride=2)
        self.conv = ConvBlock_1D(in_channels, 512, kernel_size=1)
#         self.fc1 = nn.Linear(2048, 1024)
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
#         print(x, x.shape)
        x = self.pool(x)
        x = self.conv(x)
        x = x.reshape(x.shape[0], -1)
#         print(x.shape)

        x = ActiveBen()(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
    
    
    
    
    
class InceptionV1_1D(nn.Module):
    def __init__(self, aux_logits=True, num_classes=1000):
        super(InceptionV1_1D, self).__init__()
        self.aux_logits = aux_logits
        self.conv1 = ConvBlock_1D(in_channels=1, out_chanels=64,kernel_size=7,stride=2,padding=3)
        self.conv2 = ConvBlock_1D(64, 192, kernel_size=3, stride=1, padding=1)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.inception3a = InceptionBlock_1D(192, 64, 96, 128, 16, 32, 32)
        self.inception3b = InceptionBlock_1D(256, 128, 128, 192, 32, 96, 64)
        self.inception4a = InceptionBlock_1D(480, 192, 96, 208, 16, 48, 64)
        self.inception4b = InceptionBlock_1D(512, 160, 112, 224, 24, 64, 64)
        self.inception4c = InceptionBlock_1D(512, 128, 128, 256, 24, 64, 64)
        self.inception4d = InceptionBlock_1D(512, 112, 144, 288, 32, 64, 64)
        self.inception4e = InceptionBlock_1D(528, 256, 160, 320, 32, 128, 128)
        self.inception5a = InceptionBlock_1D(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionBlock_1D(832, 384, 192, 384, 48, 128, 128)
        self.avgpool = nn.AvgPool1d(kernel_size=1, stride=1)
#         self.avgpool = nn.AvgPool3d(kernel_size=7, stride=1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(1024*4, num_classes)
#         self.fc = nn.Linear(1024, num_classes)
        
        if self.aux_logits:
            self.aux1 = InceptionAux_1D(512, num_classes)
            self.aux2 = InceptionAux_1D(528, num_classes)
        else:
            self.aux1 = self.aux2 = None
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.conv2(x)
        x = self.maxpool(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)
        x = self.inception4a(x)
#         print(x.shape)
        
        if self.aux_logits and self.training:
            aux1 = self.aux1(x)
        
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
#         print(x.shape)
        
        if self.aux_logits and self.training:
            aux2 = self.aux2(x)
        
        x = self.inception4e(x)
        x = self.maxpool(x)
        x = self.inception5a(x)

        x = self.inception5b(x)
#         print(x.shape)
        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        x = self.fc(x)
        
        if self.aux_logits and self.training:
            return aux1, aux2, x
        return x
