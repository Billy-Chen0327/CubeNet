import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BatchTraceNorm3d(nn.Module):
    def __init__(self,channel_num):
        super().__init__();
        shape = (1,channel_num,1,1,1);
        self.gamma = nn.Parameter(torch.ones(shape));
        self.beta = nn.Parameter(torch.zeros(shape));
       
    def TraceNorm3d(self,is_training,X,gamma,beta,eps,momentum):
        
        assert len(X.shape) == 5;
        mean = X.mean(dim=4,keepdim=True);
        var = ((X-mean)**2).mean(dim=4,keepdim=True);
        X_hat = (X-mean) / torch.sqrt(var+eps);
        Y = gamma*X_hat + beta;
        return Y;
    
    def forward(self,X):
        Y = self.TraceNorm3d(self.training,X,self.gamma,self.beta,eps=1e-5,momentum=0.9);
        return Y;

class DoubleConv(nn.Module):
    
    def __init__(self, in_channels, out_channels , kernel_size=np.array([1,1,7])):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,bias=False),
            BatchTraceNorm3d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size, padding=kernel_size//2,bias=False),
            BatchTraceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        );
 
    def forward(self, x):
        return self.double_conv(x);
    
class SingleConv(nn.Module):
    
    def __init__(self, in_channels,out_channels,kernel_size=np.array([3,3,7])):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size, padding=kernel_size//2,bias=False),
            BatchTraceNorm3d(out_channels),
            nn.ReLU(inplace=True),
        )
 
    def forward(self, x):
        return self.double_conv(x)    
    
class Down(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride_size=np.array([1,1,4]),kernel_size=np.array([1,1,7])):
        super().__init__();
        self.maxpool_conv = nn.Sequential(
            nn.Conv3d(in_channels,in_channels,kernel_size = kernel_size, stride=stride_size,padding=kernel_size//2,bias=False),
            BatchTraceNorm3d(in_channels),
            nn.ReLU(inplace=True),
            SingleConv(in_channels,out_channels)
            );
        
    def forward(self,x):
        return self.maxpool_conv(x);
    
class Up(nn.Module):
    
    def __init__(self,in_channels,out_channels,stride_size=np.array([1,1,4]),kernel_size=np.array([1,1,7])):
        super().__init__();
        
        self.up = nn.ConvTranspose3d(in_channels,out_channels, kernel_size=kernel_size, stride=stride_size,bias=False);
        self.norm = BatchTraceNorm3d(out_channels);
        self.act = nn.ReLU();
        self.conv = SingleConv(2*out_channels,out_channels);
        
    def forward(self,x1,x2):
        x1 = self.up(x1);
        x1 = self.norm(x1);
        x1 = self.act(x1);
        
        diffX = torch.tensor([x2.size()[4] - x1.size()[4]])
        
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2]);

        x = torch.cat([x2,x1],dim=1);
        return self.conv(x)

class FinalConv(nn.Module):
    
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.OneByOne_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=1),
            nn.Softmax(dim=1)
        )
 
    def forward(self, x):
        return self.OneByOne_conv(x);
    
class UNet(nn.Module):
    def __init__(self,n_channels=3,n_classes=3):
        super().__init__();
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        
        self.inc = DoubleConv(n_channels,8);
        self.down1 = Down(8,11);
        self.down2 = Down(11,16);
        self.down3 = Down(16,22);
        self.down4 = Down(22,32);
        self.up1 = Up(32,22);
        self.up2 = Up(22,16);
        self.up3 = Up(16,11);
        self.up4 = Up(11,8);
        self.outc = FinalConv(8,n_classes);
        
    def forward(self,x):
        x1 = self.inc(x);
        x2 = self.down1(x1);
        x3 = self.down2(x2);
        x4 = self.down3(x3);
        x5 = self.down4(x4);
        x = self.up1(x5,x4);
        x = self.up2(x,x3);
        x = self.up3(x,x2);
        x = self.up4(x,x1);
        logits = self.outc(x);
        return logits;