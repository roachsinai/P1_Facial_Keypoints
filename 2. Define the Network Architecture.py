import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# CUDA for PyTorch
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
# cudnn.benchmark = True

from models import AlexNet
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from data_load import FacialKeypointsDataset
from data_load import Rescale, RandomCrop, Normalize, ToTensor
import torch.optim as optim


best_model_epoch = 0
def train_net(n_epochs):

    # prepare the net for training
    net.train()
    
    minimum_loss = float("inf")
    best_model_epochs = []

    for epoch in range(n_epochs):  # loop over the dataset multiple times
        
        epoch_loss = 0.0
        running_loss = 0.0

        # train on batches of data, assumes you already have train_loader
        for batch_i, data in enumerate(train_loader):
            # get the input images and their corresponding labels
            images = data['image'].to(device)
            key_pts = data['keypoints'].to(device)

            # flatten pts
            key_pts = key_pts.view(key_pts.size(0), -1).to(device)

            # convert variables to floats for regression loss
            key_pts = key_pts.type(torch.FloatTensor).to(device)
            images = images.type(torch.FloatTensor).to(device)

            # forward pass to get outputs
            output_pts = net(images).to(device)

            # calculate the loss between predicted and target keypoints
            loss = criterion(output_pts, key_pts)

            # zero the parameter (weight) gradients
            optimizer.zero_grad()

            # backward pass to calculate the weight gradients
            loss.backward()

            # update the weights
            optimizer.step()

            # print loss statistics
            # to convert loss into a scalar and add it to the running_loss, use .item()
            running_loss += loss.item()
            epoch_loss += loss.item()
            if batch_i % 40 == 39:    # print every 40 batches
                print('Epoch: {}, Batch: {}, Avg. Loss: {}'.format(epoch + 1, batch_i+1, running_loss/40))
                running_loss = 0.0

        if epoch_loss < minimum_loss:
            minimum_loss = epoch_loss
            out_path = 'saved_models/{}_keypoints_model.pt'.format(normalize.norm_method)
            torch.save(net.state_dict(), out_path)
            best_model_epochs.append(epoch + 1)
            

    print('\nFinished Training, epochs(start from 1) got the minimum loss:\n    {}'.format(best_model_epochs))

if __name__ == '__main__':
    net = AlexNet()
    net.cuda()
    print(net)

    
    normalize = Normalize("CENTER", 227)
    train_data_transform = transforms.Compose([Rescale(250),\
                                               RandomCrop(227),\
                                               normalize,\
                                               ToTensor()])
    train_dataset = FacialKeypointsDataset(csv_file='data/training_frames_keypoints.csv',
                                                 root_dir='data/training/',
                                                 transform=train_data_transform)
    print('Number of images: ', len(train_dataset))


    batch_size = 10
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=4)

    criterion = nn.SmoothL1Loss()
    optimizer = optim.Adam(net.parameters(), lr=.001, betas=(.9, .999), eps=1e-08)

    n_epochs = 50 # start small, and increase when you've decided on your model structure and hyperparams
    train_net(n_epochs)
