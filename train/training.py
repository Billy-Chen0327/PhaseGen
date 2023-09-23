import copy
import torch
import config
import utils as ut
import numpy as np
import PhaseGen as model
from torch.utils.data import DataLoader

def main(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim):
    
    if torch.cuda.device_count() >= 1:
        device = torch.device('cuda');
    else:
        device = torch.device('cpu');
    
    loss_D = ut.DiscriminatorLoss();
    loss_GP = ut.GradientPenalty();
    loss_G = ut.GeneratorLoss();
    ut.net_init(net_D); ut.net_init(net_G);    
        
    net_D,net_G = net_D.to(device), net_G.to(device);
    
    trainer_D = torch.optim.Adam(net_D.parameters(),lr=lr_D,betas=(0.5,0.9));
    trainer_G = torch.optim.Adam(net_G.parameters(),lr=lr_G,betas=(0.5,0.9));
    
    loss_list = [];
    for epoch in range(1, num_epochs + 1):
        
        # Train one epoch
        metric = ut.Accumulator(4); # loss_D, loss_G, iter_num
        for x,tt in data_iter:
            # Train code
            batch_size = x.shape[0];
            z = torch.normal(0,1,size=(batch_size,1,latent_dim));
            fake_tt = copy.deepcopy(tt);
            x,tt,z,fake_tt = x.to(device), tt.to(device), z.to(device), fake_tt.to(device);
            lD,W_dis = ut.update_D(x, z, tt, fake_tt, net_D, net_G, loss_D, loss_GP, trainer_D, device);
            lG = ut.update_G(x, z, fake_tt, net_D, net_G, loss_G, trainer_G);
            metric.add(lD,lG,W_dis,1);
        
        # Show generated examples
        sta_loss_D, sta_loss_G, sta_Wdis = metric[0]/metric[-1], metric[1]/metric[-1], metric[2]/metric[-1];
        if epoch % 500 == 0:
            torch.save(net_G.state_dict(),f'./model/netG_epoch{epoch}.pt');
            torch.save(net_D.state_dict(),f'./model/netD_epoch{epoch}.pt');
        loss_list.append([sta_loss_D,sta_loss_G,sta_Wdis]);
    
    np.save('./loss.npy',np.array(loss_list));
        
if __name__ == '__main__':
    
    settings = config.settings;

    seed = settings['seed'];
    torch.manual_seed(seed);
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed);
        torch.cuda.manual_seed_all(seed);
    np.random.seed(seed);
    torch.backends.cudnn.benchmark = False;
    torch.backends.cudnn.deterministic = True;
    
    data_path = './data';
    net_D = model.net_D(); net_G = model.net_G();
    net_D.train(); net_G.train();
    
    lr_D, lr_G, latent_dim, num_epochs = settings['lr_D'], settings['lr_G'], settings['target_len'], settings['epoch'];
    dataset = ut.getGen(data_path);
    data_iter = DataLoader(dataset=dataset,
                            batch_size=settings['batch_size'],
                            shuffle=True,
                            num_workers=settings['num_workers']);
    main(net_D, net_G, data_iter, num_epochs, lr_D, lr_G, latent_dim);
