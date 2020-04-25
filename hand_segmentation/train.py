import sys
import os
from os.path import join
from optparse import OptionParser
import numpy as np

import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

from torchvision import transforms

import matplotlib.pyplot as plt

from model import UNet
from dataloader import DataLoader

import cv2

def train_net(net,
              epochs=100,
              batch_size=10,
              data_dir='../../egohands_data/',
              n_classes=2,
              lr=0.0001,
              gpu=False):
    loader = DataLoader(data_dir, batch_size, input_width=240, input_height=240)

    # optimizer = optim.SGD(net.parameters(),
    #                       lr=lr,
    #                       momentum=0.99,
    #                       weight_decay=0.0005)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=0.0001)

    sch = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, threshold=0.005)

    # criterion = nn.CrossEntropyLoss()
    # criterion = nn.BCEWithLogitsLoss()
    criterion = nn.BCELoss()

    results_file = open('Train.csv', 'w')
    results_file.close()
    results_file = open('Eval.csv', 'w')
    results_file.close()

    min_train_loss = 0
    min_eval_loss = 0
    for epoch in range(epochs):
        print('Epoch %d/%d' % (epoch + 1, epochs))
        print('Training...')
        net.train()
        loader.setMode('train')
        loader.shuffle_data()

        epoch_loss = 0
        for i, (data, targets) in enumerate(loader, 1):
            if gpu:
                data = data.cuda()
                targets = targets.cuda()

            optimizer.zero_grad()
            predictions = net(data)
            loss = criterion(predictions, targets)

            epoch_loss += loss.item()

            print('Training batch %d - Loss: %.6f' % (i, loss.item()))

            # optimize weights
            loss.backward()
            optimizer.step()

        train_loss = epoch_loss / i
        print('Epoch %d Training Loss: %.6f' % (epoch+1, train_loss))
        
        with open('Train.csv', 'a') as results_file:
            results_file.write('%d,%f\n'%(epoch+1, train_loss))

        if epoch == 0 or min_train_loss >= train_loss:
            min_train_loss = train_loss
            print('Saving best network..')
            torch.save(net.state_dict(), 'best_train_network.pth')
            # torch.save(net, 'best_train_network.pth')

        if epoch % 10 == 9:
            print('Saving network..')
            torch.save(net.state_dict(), 'network.pth')
            # torch.save(net, 'network.pth')

        torch.cuda.empty_cache()
        
        print('Evaluating...')
        net.eval()
        loader.setMode('eval')
        loader.shuffle_eval()

        epoch_loss = 0
        for i, (data, targets) in enumerate(loader, 1):
            if gpu:
                data = data.cuda()
                targets = targets.cuda()

            # print(data.size(), targets.size())

            predictions = net(data)
            loss = criterion(predictions, targets)

            epoch_loss += loss.item()

        val_loss = epoch_loss / i
        sch.step(val_loss)

        image = data[0].cpu().detach().numpy().transpose((1, 2, 0))*255.
        
        # pred = predictions[0]
        # print(pred.size())
        # pred_sm = F.softmax(pred, dim=0)
        # print(pred_sm.size())
        # _,pred_label = torch.max(pred_sm, dim=0)
        # print(pred_label.size())
        # mask = pred_label.cpu().detach().numpy().squeeze()*255.
        pred = predictions[0]
        # pred_label = torch.sigmoid(pred).round()
        pred_label = pred.round()
        mask = pred_label.cpu().detach().numpy().astype('uint8').squeeze()*255.

        cv2.imwrite('eval_results/'+'epoch_%d_img.jpg'%(epoch+1), image)
        cv2.imwrite('eval_results/'+'epoch_%d_tgt.jpg'%(epoch+1), mask)

        print('Epoch %d Evaluation Loss: %.6f' % (epoch+1, val_loss))

        with open('Eval.csv', 'a') as results_file:
            results_file.write('%d,%f\n'%(epoch+1,epoch_loss / i)) 
            
        if epoch == 0 or min_eval_loss >= val_loss:
            min_eval_loss = val_loss
            print('Saving best network..')
            torch.save(net.state_dict(), 'best_eval_network.pth')
            # torch.save(net, 'best_eval_network.pth')

            if train_loss > val_loss:
                torch.save(net.state_dict(), 'best_network.pth')
                # torch.save(net, 'best_network.pth')

            # displays test images with original and predicted masks after training
            loader.setMode('test')
            net.eval()
            with torch.no_grad():
              for _, (img, filename) in enumerate(loader):
                  test_network(net, img, filename, gpu)

    # loader.setMode('test')
    # # net.load_state_dict(torch.load('best_eval_network.pth'))
    # net.eval()
    # with torch.no_grad():
    #   for _, (img, filename) in enumerate(loader):
    #       test_network(net, img, filename, gpu)

def test_net(net, data_dir='../../egohands_data/', gpu=False):
    loader = DataLoader(data_dir, 1, input_width=160, input_height=90)
    loader.setMode('test')
    net.load_state_dict(torch.load('best_eval_network.pth'))
    net.eval()
    with torch.no_grad():
      for _, (img, filename) in enumerate(loader):
          test_network(net, img, filename, gpu)

def test_network(net, img_torch, filename, gpu):
    # print(filename)
    if gpu:
        img_torch = img_torch.cuda()
    pred = net(img_torch)
    # pred_sm = softmax(pred)
    # _,pred_label = torch.max(pred_sm,1)
    pred_label = pred.round()

    # plt.subplot(1, 3, 1)
    # plt.imshow(image*255.)
    # plt.subplot(1, 3, 2)
    # plt.imshow(label*255.)
    # plt.subplot(1, 3, 3)
    # plt.imshow(pred_label.cpu().detach().numpy().squeeze()*255.)
    # plt.show()
    mask = pred_label.cpu().detach().numpy().astype('uint8').squeeze()*255.
    # print(mask.shape)
    tokens = filename.split('/')
    if not os.path.isdir('test_results/' + tokens[-2]):
        os.mkdir('test_results/' + tokens[-2])
    cv2.imwrite('test_results/' + tokens[-2] + '/' + tokens[-1], mask)
    
def get_args():
    parser = OptionParser()
    parser.add_option('-e', '--epochs', dest='epochs', default=100, type='int', help='number of epochs')
    parser.add_option('-b', '--batch-size', dest='batch_size', default=10, type='int', help='batch size')
    parser.add_option('-L', '--learning-rate', dest='lr', default=0.0001, type='float', help='learning rate')
    parser.add_option('-c', '--n-classes', dest='n_classes', default=2, type='int', help='number of classes')
    parser.add_option('-d', '--data-dir', dest='data_dir', default='../../egohands_data/', help='data directory')
    parser.add_option('-g', '--gpu', action='store_true', dest='gpu', default=False, help='use cuda')
    parser.add_option('-l', '--load', dest='load', default=False, help='load file model')

    (options, args) = parser.parse_args()
    return options

if __name__ == '__main__':
    args = get_args()

    net = UNet()
    print(net)

    if not os.path.isdir('eval_results'):
        os.mkdir('eval_results')

    if not os.path.isdir('test_results'):
        os.mkdir('test_results')

    if args.load:
        net.load_state_dict(torch.load(args.load))
        print('Model loaded from %s' % (args.load))

    if args.gpu:
        net = net.cuda()
        cudnn.benchmark = True

    train_net(net=net,
        epochs=args.epochs,
        n_classes=args.n_classes,
        gpu=args.gpu,
        data_dir=args.data_dir, 
        lr=args.lr,
        batch_size=args.batch_size)
    # test_net(net, gpu=True)
