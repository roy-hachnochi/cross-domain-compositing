"""Training scripts for D2IMNet model."""
import os, math
import numpy as np
import torch

import SVR.utils as utils
from SVR.models.D2IM import D2IM_Net
from SVR.datasets.SDFDataset import SDFDataset as Dataset

def test_one_with_gttransmat(model, dataset, gridworldcoords, cat_id, shape_id, cam_id, output_dir, config):
    """Test one image model pair with ground truth transfromation matrix."""
    rgb_image, _, transmat = dataset.get_testdata(cat_id, shape_id, cam_id)
    if((rgb_image is None) or (transmat is None)):
        return 
    if(config.cuda):
        rgb_image, transmat = rgb_image.cuda(), transmat.cuda()
    rgba_image = rgb_image.unsqueeze(0)
    transmat = transmat.unsqueeze(0)      

    # predict per-point sdfs
    test_pointnum = config.test_pointnum
    times = math.ceil(gridworldcoords.size(1)/test_pointnum)
    pred_value_list = []
    for i in range(times):
        endind = min((i+1)*test_pointnum, gridworldcoords.size(1))
        points_batch = gridworldcoords[:,i*test_pointnum:endind,:]
        pred_gtvalues= model.inference_sdf_with_detail(points_batch, rgba_image, transmat)
        pred_value_list.append(pred_gtvalues)
    pred_values = torch.cat(pred_value_list, dim=1)
    
    # reshape the implicit field
    pred_values = pred_values.view(config.mcube_znum, config.mcube_znum,config.mcube_znum)
    pred_values = pred_values.cpu().detach().numpy()
    pred_values = pred_values/10.0    

    # extract the iso-surface
    if(not os.path.exists(output_dir + '/' + cat_id)):
        os.makedirs(output_dir + '/' + cat_id)
    ply_fname = output_dir + '/' + cat_id + '/' + shape_id + '.ply'
    #utils.render_grid_occupancy(image_fname, pred_values)
    utils.render_implicits(ply_fname, pred_values)
    
    
def test(epoch, model, dataset, config, testlist):
    output_dir = config.output_dir+'/epoch_'+str(epoch+1)
    if(not os.path.exists(output_dir)):
        os.makedirs(output_dir)
    with torch.no_grad():
        gridworldcoords = utils.sample_grid_points(config.mcube_znum, config.mcube_znum, config.mcube_znum) # sampled gridpoints under camera's view
        gridworldcoords = torch.tensor(gridworldcoords).float().view(-1,3).unsqueeze(0)
        if(config.cuda):
            gridworldcoords = gridworldcoords.cuda()
        for testdata in testlist:
            cat_id = testdata['cat_id']
            shape_id = testdata['shape_id']
            # select the view with most details
            cam_id = 33
            test_one_with_gttransmat(model, dataset, gridworldcoords, cat_id, shape_id, cam_id, output_dir, config)              


def train_epoch(epoch, model, optimizer, data_iter, config):
    losslist = []
    for batch_idx, batch in enumerate(data_iter):
        # train
        [points, values, gradients, mcimage, transmat, scale] = batch
        if(config.cuda):
            points, values, gradients, mcimage, transmat, scale = points.cuda(), values.cuda(), gradients.cuda(), mcimage.cuda(), transmat.cuda(), scale.cuda()

        Baseloss, SDFloss, Laploss = model(points, values, gradients, mcimage, transmat, scale)
        loss = Baseloss + SDFloss + Laploss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss = loss.detach()
        losslist.append(loss.unsqueeze(0))

        # print log
        if((batch_idx+1)%config.plot_every_batch==0 or batch_idx==len(data_iter)-1):
            # plot the loss
            logline = 'epoch:%d//%d, batch:%d//%d, base_loss:%f, sdf_loss:%f, lap_loss:%f, total_loss:%f' % (epoch+1, config.epochs, batch_idx+1, len(data_iter), Baseloss.item(), SDFloss.item(), Laploss.item(), loss.item())
            print(logline)
            utils.print_log(config.log, logline)

    # save the trained model
    if((epoch+1)%config.save_every_epoch==0):
        utils.save_checkpoint(epoch, model, optimizer, loss.item(), config.model_dir+'/model'+str(epoch+1)+'.pt.tar')

    loss = torch.sum(torch.cat(losslist))/len(losslist)
    return loss


def train(config):
    """Training function."""
    if(config.cuda):
        torch.cuda.set_device(config.gpu)
        torch.backends.cudnn.benchmark = True

    # Specify model network
    model = D2IM_Net(config)
    if(config.cuda):
        model.cuda()
    
    # Load training set
    trainset = Dataset(config, 'train')
    train_iter = torch.utils.data.DataLoader(trainset, batch_size=config.train_batch_size, shuffle=True,num_workers=8)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.lr, betas=(config.beta1, 0.999), weight_decay=1e-5)
    
    epoch = 0
    best_test_loss = 1000000
    # If pretrained model exists, load pretrained model. Resume training if the input epoch is higher than the existing epoch
    if(config.load_pretrain and os.path.exists(config.model_dir+'/' + config.model_folder_name  + config.sdf_model_name)):
        epoch, model, optimizer, best_test_loss = utils.load_checkpoint(config.model_dir+'/'+ config.model_folder_name + config.sdf_model_name,model, optimizer)
        print('pretrained model loaded')
    else:
        f = open(config.log, 'w')
        f.write('')
        f.close()
 
    # training
    while(epoch<config.epochs):
        # train
        model.train()
        loss = train_epoch(epoch, model, optimizer, train_iter, config)
        print('epoch %d finished.'%(epoch+1))
        if(best_test_loss>loss):
            # save it as the best model
            best_test_loss = loss
            utils.save_checkpoint(epoch, model, optimizer, loss.item(), config.model_dir+'/'+config.model_folder_name+config.sdf_model_name)
            
        epoch += 1


if __name__ == "__main__":
    config = utils.get_args()
    config.load_pretrain = True
    os.makedirs(config.model_dir+'/' + config.model_folder_name, exist_ok=True)
    train(config)
                
