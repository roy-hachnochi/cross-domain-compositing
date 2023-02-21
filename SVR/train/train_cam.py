import os
import numpy as np 
import torch 
import torch.nn as nn
from SVR.datasets.CamDataset import CamDataset
from SVR.models.CamNet import CamNet as Net
import SVR.utils as utils

def test(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
    model = Net(config)
    if(config.cuda):
        model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.cam_lr, betas=(config.beta1, 0.999), weight_decay=1e-5)
    
    testset = CamDataset(config, 'test')
    test_iter = torch.utils.data.DataLoader(testset, batch_size=config.cam_batch_size, shuffle=False)
    epoch, model, optimizer = utils.load_checkpoint(config.model_dir+'/'+config.model_folder_name+config.cam_model_name,model, optimizer)
    output_dir = config.output_dir+str(epoch)+'_epoch/'
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)

    for batch_idx, batch in enumerate(test_iter):
        [images, cams, sampling] = batch
        if(config.cuda):
            images, cams, sampling = images.cuda(), cams.cuda(), sampling.cuda()
        pred_cams, _ = model(images, cams, sampling)
        # output

def train(config):
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
    
    model = Net(config)
    mseloss = nn.MSELoss()
    if(config.cuda):
        model.cuda()
        mseloss.cuda()
    
    trainset = CamDataset(config, 'train')
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=config.cam_batch_size, shuffle=True, drop_last = True)

    optimizer = torch.optim.Adam(model.parameters(), lr=config.cam_lr, betas=(config.beta1, 0.999))
    epoch = 0

    if(config.load_pretrain and os.path.exists(config.model_dir+'/'+config.model_folder_name + config.cam_model_name)):
        epoch, model, optimizer, best_test_loss = utils.load_checkpoint(config.model_dir+'/' + config.model_folder_name + config.cam_model_name,model, optimizer)
    else:
        f = open(config.log, 'w')
        f.write('')
        f.close()

    # train
    while(epoch<config.epochs):
        model.train()
        for batch_idx, batch in enumerate(train_iter):
            [images, cams, sampling] = batch
            if(config.cuda):
                images, cams, sampling = images.cuda(), cams.cuda(), sampling.cuda()
            _, loss = model(images, cams, sampling)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach()

            # print
            logline = 'epoch:%d//%d, batch:%d//%d, loss:%f' % (epoch+1, config.epochs, batch_idx+1, len(train_iter), loss.item())
            print(logline)
            utils.print_log(config.log, logline)

        utils.save_checkpoint(epoch, model, optimizer, loss.item(), config.model_dir+'/'+config.model_folder_name+config.cam_model_name)
        if((epoch+1)%config.save_every_epoch==0):
            utils.save_checkpoint(epoch, model, optimizer, loss.item(), config.model_dir+'/model'+str(epoch+1)+'.pt.tar')
        
        epoch += 1
    
if __name__ == "__main__":
    config = utils.get_args()
    config.load_pretrain = True
    os.makedirs(config.model_dir+'/' + config.model_folder_name, exist_ok=True)
    train(config)