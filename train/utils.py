import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset

def net_init(net):
    for name,param in net.named_parameters():
        nn.init.normal_(param,0.0,0.02);
        
class getGen(Dataset):
    
    def __init__(self,npz_path):
        
        self.npz_path = npz_path;
        self.npz_list = os.listdir(npz_path);
        self.sample_num = len(self.npz_list);
        
    def __getitem__(self,index):
        
        npz_name = self.npz_list[index];
        data_dict = np.load(os.path.join(self.npz_path,npz_name));
        tp = data_dict['Parr']; ts = data_dict['Sarr'];
        tr = torch.Tensor(data_dict['data']);
        tt = torch.Tensor([ts-tp]).unsqueeze(0);
        return tr,tt;
        
    def __len__(self):
        
        return self.sample_num;

class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super().__init__();
    def forward(self,f_real,f_fake):
        return (-f_real).mean(), f_fake.mean();

class GeneratorLoss(nn.Module):
    def __init__(self):
        super().__init__();
    def forward(self,f_fake):
        return -f_fake.mean();

class GradientPenalty(nn.Module):
    def __init__(self):
        super().__init__();
    def forward(self,x_real,tt_real,x_fake,tt_fake,net_D,device):
        batch_size = x_real.shape[0];
        alpha = torch.rand(batch_size,1,1,device=device);
        alpha_x = alpha.expand(x_real.size());
        x = alpha_x*x_real + ((1-alpha_x)*x_fake);
        alpha_tt = alpha.expand(tt_real.size());
        tt = alpha_tt*tt_real + ((1-alpha_tt)*tt_fake);
        x = torch.autograd.Variable(x,requires_grad=True);
        f = net_D(x,tt);
        gradients, *_ = torch.autograd.grad(outputs=f,
                                            inputs=x,
                                            grad_outputs=f.new_ones(f.shape),
                                            create_graph=True);
        gradients = gradients.reshape(batch_size,-1);
        norm = gradients.norm(2,dim=-1);
        return torch.mean((norm-1)**2);

class Accumulator:
    
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def update_D(x,z,tt,fake_tt,net_D,net_G,loss,loss_GP,trainer_D,device):
    
    ''' Update discriminator '''
    trainer_D.zero_grad();
    real_y = net_D(x,tt);
    fake_x = net_G(z,fake_tt);
    fake_y = net_D(fake_x.detach(),fake_tt.detach());
    
    loss_real,loss_fake = loss(real_y,fake_y);
    GPloss = loss_GP(x,tt,fake_x,fake_tt,net_D,device);
    
    loss_D = loss_real + loss_fake + GPloss;
    loss_D.backward();
    trainer_D.step();
    W_dis = (loss_real + loss_fake).item()
    
    return loss_D,W_dis;

def update_G(x,z,fake_tt,net_D,net_G,loss,trainer_G):
    
    ''' Update generator '''
    trainer_G.zero_grad();
    fake_x = net_G(z,fake_tt);
    fake_y = net_D(fake_x,fake_tt);
    loss_G = loss(fake_y);
    loss_G.backward(); trainer_G.step();
    return loss_G;