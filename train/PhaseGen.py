import torch
import torch.nn as nn
from collections import OrderedDict

class Residual_Block(nn.Module):
    
    def __init__(self,channels):
        
        super().__init__();
        self.conv1 = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size=9, padding = 4);
        self.norm1 = nn.InstanceNorm1d(channels,affine=True);
        self.prelu = nn.LeakyReLU(0.2);
        self.conv2 = nn.Conv1d(in_channels = channels, out_channels = channels, kernel_size=9, padding = 4);
        self.norm2 = nn.InstanceNorm1d(channels,affine=True);
        
    def forward(self,x):
        
        residual = self.conv1(x);
        residual = self.norm1(residual);
        residual = self.prelu(residual);
        residual = self.conv2(residual);
        residual = self.norm2(residual);
        
        return x + residual;
    
class branch_block(nn.Module):
    
    def __init__(self,in_channels,out_channels):
        
        super().__init__();
        self.block1 = Residual_Block(in_channels);
        self.block2 = Residual_Block(in_channels);
        self.block3 = Residual_Block(in_channels);
        self.block4 = Residual_Block(in_channels);
        self.block5 = Residual_Block(in_channels);
        self.block6 = Residual_Block(in_channels);
        self.block7 = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=1);
    
    def forward(self,x):
        
        y = self.block1(x);
        y = self.block2(y);
        y = self.block3(y);
        y = self.block4(y);
        y = self.block5(y);
        y = self.block6(y);
        y = self.block7(y);
        
        return y;
    
class net_G(nn.Module):
    
    def __init__(self):
        super().__init__();
        
        self.tt_embed = nn.Sequential(OrderedDict([
            ('linear',nn.Linear(1,3000,bias=False)),
            ('conv',nn.Conv1d(1,out_channels=64,kernel_size=9,padding=4)),
            ('act',nn.LeakyReLU(0.2),)
            ]));
        
        self.z_embed = nn.Sequential(OrderedDict([
            ('conv',nn.Conv1d(1,out_channels=64,kernel_size=9,padding=4)),
            ('act',nn.LeakyReLU(0.2),),
            ]));
        
        self.ResConv_blockX = branch_block(128,1);
        self.ResConv_blockY = branch_block(128,1);
        self.ResConv_blockZ = branch_block(128,1);        
    
    def forward(self, z, tt):
        
        latent_tt = self.tt_embed(tt);
        latent_z = self.z_embed(z);
        compX = self.ResConv_blockX(torch.cat([latent_tt,latent_z],dim=1));
        compY = self.ResConv_blockY(torch.cat([latent_tt,latent_z],dim=1));
        compZ = self.ResConv_blockZ(torch.cat([latent_tt,latent_z],dim=1));
        out = torch.cat([compX,compY,compZ],dim=1);
        
        return out;
    
class Conv_D(nn.Module):
    def __init__(self,in_channels,out_channels,kernel_size,padding,stride):
        super().__init__();
        self.conv = nn.Conv1d(in_channels=in_channels,out_channels=out_channels,kernel_size=kernel_size,padding=padding,stride=stride);
        self.norm = nn.InstanceNorm1d(out_channels,affine=True);
        self.act = nn.LeakyReLU(0.2);
        
    def forward(self,x):
        x = self.conv(x);
        x = self.norm(x);
        x = self.act(x);
        return x;
        
class net_D(nn.Module):
    
    def __init__(self,in_channels=3):
        super().__init__();
        self.tt_embed = nn.Sequential(OrderedDict([
            ('linear',nn.Linear(1,3000,bias=False)),
            ('conv',nn.Conv1d(1,out_channels=64,kernel_size=9,padding=4)),
            ('act',nn.LeakyReLU(0.2)),
            ]));
        self.x_embed = nn.Sequential(OrderedDict([
            ('conv',nn.Conv1d(in_channels,out_channels=64,kernel_size=9,padding=4)),
            ('act',nn.LeakyReLU(0.2)),
            ]));
        channels = [128,256,512,1024];
        strides = [3,3,3];
        block_seq = OrderedDict([('SubBlock'+str(i+1),Conv_D(in_channels=channels[i],out_channels=channels[i+1],kernel_size=5,padding=3,stride=strides[i])) for i in range(len(strides))]);
        self.dig_block = nn.Sequential(block_seq);
        self.flat_block = nn.Sequential(OrderedDict([
            ('GlobalAvgPool',nn.AdaptiveMaxPool1d(1)),
            ('1x1_conv',nn.Conv1d(channels[-1],1024,kernel_size=1)),
            ('act',nn.LeakyReLU(0.2)),
            ('flat_conv',nn.Conv1d(1024,1,kernel_size=1)),
            ]));
    
    def forward(self,x,tt):
        
        latent_tt = self.tt_embed(tt);
        latent_x = self.x_embed(x);
        x = self.dig_block(torch.cat([latent_tt,latent_x],dim=1));        
        x = self.flat_block(x);
        
        return x;
    
    