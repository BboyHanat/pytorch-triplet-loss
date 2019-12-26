"""
Name : train.py
Author  : Hanat
Contect : hanati@tezign.com
Time    : 2019-12-24 14:34
Desc:
"""



from configs import conf
from networks.triplet_loss import TripletLoss
from networks.model_invoke import NetWorkInvoker
from utils.dataloader import TripletDataset

import os
import torch
import torch.optim as optim
from torch.utils.data import DataLoader





def train(train_data_path, valid_data_path):
    use_cuda = conf['train_gpu_config']['use_cuda']
    img_size = conf['train_parameter']['img_size']

    batch_size = conf['train_parameter']['batch_size']
    epoch = conf['train_parameter']['epoch']
    learning_rate = conf['train_parameter']['learning_rate']
    val_interval_step = conf['train_parameter']['val_interval_step']
    valid_iter_num = conf['train_parameter']['valid_iter_num']
    gpu_enum = [gpu['gpu_enum'] for gpu in conf['train_gpu_config']['gpu_enum']]
    format_list = [gpu['format_list'] for gpu in conf['train_gpu_config']['format_list']]
    embedding = conf['train_parameter']['embedding']
    model_name = conf['train_parameter']['model_name']
    pretrained = conf['train_parameter']['pretrained']

    net = NetWorkInvoker(model_name=model_name, embedding=embedding, pretrained=pretrained)
    optimizer = optim.Adadelta(net.parameters(), lr=learning_rate)
    loss_triplet = TripletLoss(margin=0.5)

    if use_cuda and len(gpu_enum) > 1:
        net = torch.nn.DataParallel(net, device_ids=gpu_enum)  ##转换为多GPU训练模型
        loss_triplet = loss_triplet.cuda()#device=gpu_enum[0])

    elif use_cuda:
        net = net.cuda()
        loss_triplet = loss_triplet.cuda()

    data_gen_train = TripletDataset(train_data_path, format_list=format_list)
    data_gen_valdation = TripletDataset(train_data_path, format_list=format_list)

    train_batch_data = DataLoader(data_gen_train,
                                  batch_size=batch_size,
                                  shuffle=bool(int(conf['train_parameter']['data_shuffle'])),
                                  num_workers=int(conf['train_parameter']['data_loader_works']),
                                  )
    valid_batch_data = DataLoader(data_gen_valdation,
                                  batch_size=batch_size,
                                  shuffle=bool(int(conf['train_parameter']['data_shuffle'])),
                                  num_workers=int(conf['train_parameter']['data_loader_works']),
                                  )

    # define densenet model
    accuracy_last_time = 0.0
    net.train()
    for e in range(epoch):
        train_iter = iter(train_batch_data)
        valid_iter = iter(valid_batch_data)
        train_step = len(train_batch_data) - 1
        valid_step = len(valid_batch_data) - 1
        print('epoch:{}/{}'.format(e, epoch))
        for t in range(train_step):
            image_triplet = train_iter.next()
            if use_cuda:
                image_triplet = image_triplet.cuda()

            preds_anchor = net(image_triplet[0])
            preds_pos = net(image_triplet[1])
            preds_neg = net(image_triplet[1])
            loss = loss_triplet(preds_anchor, preds_pos, preds_neg)
            loss.backward()
            optimizer.step()
            print('epoch: {}/{}, step: {}/{}, training_loss: {}'.format(e, epoch, t, train_step, loss))




