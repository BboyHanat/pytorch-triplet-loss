import os

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from configs import conf
from utils.dataloader import TripletDataset
from networks.triplet_loss import TripletLoss
from networks.model_invoke import NetWorkInvoker


def train():
    use_gpu = conf['train_gpu_config']['use_gpu']
    img_size = conf['train_parameter']['img_size']

    batch_size = conf['train_parameter']['batch_size']
    epoch = conf['train_parameter']['epoch']
    learning_rate = conf['train_parameter']['learning_rate']

    val_interval_step = conf['train_parameter']['val_interval_step']
    valid_iter_num = conf['train_parameter']['valid_iter_num']

    format_list = [formats['format_list'] for formats in conf['train_parameter']['format_list']]
    embedding = conf['train_parameter']['embedding']
    model_name = conf['train_parameter']['model_name']

    model_save_path = conf['path_config']['model_save_path']
    pretrained_model = conf['path_config']['pretrained_model']

    train_data_path = conf['path_config']['train_data_path']
    valid_data_path = conf['path_config']['valid_data_path']

    device = torch.device("cpu")
    net = NetWorkInvoker(model_name=model_name, embedding=embedding)
    if pretrained_model:
        net.load_weight(pretrained_model, devices=device)
    loss_triplet = TripletLoss(margin=5)

    if use_gpu:
        device = torch.device("cuda:0")
        net = net.to(device)
        loss_triplet = loss_triplet.to(device)
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)

    gpu_enum = [gpu['gpu_enum'] for gpu in conf['train_gpu_config']['gpu_enum']]
    print(gpu_enum)

    if use_gpu:
        net = net.to(device)
        if len(gpu_enum) > 1:
            net = torch.nn.DataParallel(net, device_ids=gpu_enum)   # 转换为多GPU训练模型
        loss_triplet = loss_triplet.to(device)

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
            anchor_img, pos_img, neg_img = train_iter.next()
            if use_gpu:
                anchor_img = anchor_img.to(device)
                pos_img = pos_img.to(device)
                neg_img = neg_img.to(device)
            preds_anchor = net(anchor_img)
            preds_pos = net(pos_img)
            preds_neg = net(neg_img)
            net.zero_grad()

            loss = loss_triplet(preds_anchor, preds_pos, preds_neg)
            loss.backward()
            optimizer.step()
            print('epoch: {}/{}, step: {}/{}, training_loss: {} \r'.format(e, epoch, t, train_step, loss))
        if len(gpu_enum) > 1 and use_gpu:
            torch.save(net.module.state_dict(), os.path.join(model_save_path, 'resnet50_feature_model_e{}.pth'.format(e)))
        else:
            torch.save(net.state_dict(), os.path.join(model_save_path, 'resnet50_feature_model_e{}.pth'.format(e)))


train()
