
from models import KESMSalienceEstimation
from loss import pairwise_loss
import util
import os
import re
import sys

import numpy as np
import time

import torch
from torch.autograd.variable import Variable
import data_gen

config = util.config_train


def mock_iter():
    npos = 3
    nneg = 5
    L = 10
    nw = 20
    allsize = 128
    while True:
        # pos, neg, pos_desp, neg_desp, doc_W, doc_E, doc_desp = next(train_iter)
        # doc_E: [ne, esize]
        # doc_description: [ne, L, wsize]
        # doc_W: [nw, wsize]
        # pos: [npos, esize]
        # pos_description: [npos, L, wsize]
        # neg: [nneg, esize]
        # neg_description: [nneg, L, wsize]

        pos = np.random.rand(npos * allsize).reshape((npos, allsize)).astype(np.float32)
        neg = np.random.rand(nneg * allsize).reshape((nneg, allsize)).astype(np.float32)
        pos_desp = np.random.rand(npos * L * allsize).reshape((npos, L, allsize)).astype(np.float32)
        neg_desp = np.random.rand(nneg * L * allsize).reshape((nneg, L, allsize)).astype(np.float32)
        doc_E = np.concatenate((pos, neg), axis=0)
        doc_desp = np.concatenate((pos_desp, neg_desp), axis=0)
        doc_W = np.random.rand(nw * allsize).reshape((nw, allsize)).astype(np.float32)
        yield pos, neg, pos_desp, neg_desp, doc_W, doc_E, doc_desp


def get_optimizer(params, opttype, lr):
    grad_params = [param for param in params if param.requires_grad]
    if opttype == 'Adam':
        return torch.optim.Adam(grad_params, weight_decay=config['weight_decay'], lr=lr)
    elif opttype == 'SGD':
        return torch.optim.SGD(grad_params, weight_decay=config['weight_decay'], momentum=0.9, lr=lr, nesterov=False)
    else:
        raise ValueError('Optimizer not specified.')


def train():
    net = KESMSalienceEstimation(config['esize'], config['wsize'], None, config['keesize'], config['L'],
                                 config['cnn_ksize'], config['K'], config['cos_epsilon'], config['dropout_p'])

    # prepare data generator
    # data_gen.prepare('oldfile/d10000-aa', 'oldfile/div10000-aa.json', 'oldfile/maps.txt', 'oldfile/w2v_vec')
    data_gen.prepare('data_train/articles_trainset.csv', 'data_train/entities_trainset.csv', 'data_train/reformat_trainset.json', 'data_all/maps.txt', 'data_all/w2v_vec')

    if config['use_pretrained_model']:
        last_epoch, pretrained = util.last_model(config['model_dir'])
        print('Use pre-trained model <%s>' % pretrained)
        net.load_state_dict(torch.load(config['model_dir'] + '/' + pretrained))
    else:
        last_epoch = 0
    if config['gpu']:
        net.cuda()
    for param in net.parameters():
        param.requires_grad = True
    optimizer = get_optimizer(net.parameters(), config['optim'], config['lr'])
    if not os.path.isdir(config['model_dir']):
        os.mkdir(config['model_dir'])

    # train
    dec_lr = False
    net.train()
    for epoch in range(last_epoch, last_epoch + config['epochs']):
        print('epoch %d' % epoch, flush=True)
        t = time.time()

        # get data iterator for current epoch
        train_iter = data_gen.getDocumentData()

        batch = 0
        batch_loss = 0
        epoch_loss = 0
        while True:
            try:
                while True:
                    pos, neg, pos_desp, neg_desp, doc_W, doc_E, doc_desp = next(train_iter)
                    # doc_E: [ne, esize]
                    # doc_description: [ne, L, wsize]
                    # doc_W: [nw, wsize]
                    # pos: [npos, esize]
                    # pos_description: [npos, L, wsize]
                    # neg: [nneg, esize]
                    # neg_description: [nneg, L, wsize]
                    npos = pos.shape[0]
                    nneg = neg.shape[0]

                    if npos != 0 and nneg != 0:
                        break

                    del pos, neg, pos_desp, neg_desp, doc_W, doc_E, doc_desp

                pos = np.expand_dims(pos, axis=0)
                neg = np.expand_dims(neg, axis=0)
                pos_desp = np.expand_dims(pos_desp, axis=0)
                neg_desp = np.expand_dims(neg_desp, axis=0)
                # pos: [1, npos, esize]
                # pos_description: [1, npos, L, wsize]
                # neg: [1, nneg, esize]
                # neg_description: [1, nneg, L, wsize]

                pos_lis = []
                neg_lis = []
                pos_desp_lis = []
                neg_desp_lis = []
                for ipos in range(npos):
                    for ineg in range(nneg):
                        pos_lis.append(pos[:, ipos, :])
                        neg_lis.append(neg[:, ineg, :])
                        pos_desp_lis.append(pos_desp[:, ipos, :, :])
                        neg_desp_lis.append(neg_desp[:, ineg, :, :])

                pos = np.stack(pos_lis, axis=0)  # [batch, 1, esize]
                neg = np.stack(neg_lis, axis=0)  # [batch, 1, wsize]
                pos_desp = np.stack(pos_desp_lis, axis=0)  # [batch, 1, L, wsize]
                neg_desp = np.stack(neg_desp_lis, axis=0)  # [batch, 1, L, wsize]

                pos = Variable(torch.from_numpy(pos))
                neg = Variable(torch.from_numpy(neg))
                pos_desp = Variable(torch.from_numpy(pos_desp))
                neg_desp = Variable(torch.from_numpy(neg_desp))
                doc_E = Variable(torch.from_numpy(doc_E))
                doc_W = Variable(torch.from_numpy(doc_W))
                doc_desp = Variable(torch.from_numpy(doc_desp))

                if config['gpu']:
                    pos = pos.cuda()
                    neg = neg.cuda()
                    pos_desp = pos_desp.cuda()
                    neg_desp = neg_desp.cuda()
                    doc_E = doc_E.cuda()
                    doc_W = doc_W.cuda()
                    doc_desp = doc_desp.cuda()

                tp_pos = torch.split(pos, config['batch_size'], dim=0)
                tp_neg = torch.split(neg, config['batch_size'], dim=0)
                tp_pos_desp = torch.split(pos_desp, config['batch_size'], dim=0)
                tp_neg_desp = torch.split(neg_desp, config['batch_size'], dim=0)

                for row in zip(tp_pos, tp_pos_desp, tp_neg, tp_neg_desp):
                    batch += 1
                    optimizer.zero_grad()
                    inputs = row + (doc_E, doc_desp, doc_W)

                    outputs = net(*inputs)
                    loss = pairwise_loss(*outputs)
                    loss.backward()
                    optimizer.step()
                    batch_loss += float(loss)

                    if batch % 1000 == 0:
                        print('Batch %d avg loss = %.4f' % (batch, batch_loss / 1000), flush=True)
                        torch.cuda.empty_cache()
                        epoch_loss += batch_loss
                        batch_loss = 0

                del pos, neg, pos_desp, neg_desp, doc_W, doc_E, doc_desp
                del tp_pos, tp_neg, tp_pos_desp, tp_neg_desp
            except StopIteration:
                break

        print('Epoch %d avg loss = %.6f' % (epoch, epoch_loss / batch), flush=True)
        torch.save(net.state_dict(), "{}/{}_{}.pth".format(config['model_dir'], config['model_prefix'], epoch))
        print('Finish epoch %d, cost %.2f' % (epoch, time.time() - t), flush=True)

        if (epoch == config['dec_lr_epoch'] or epoch_loss / batch < config['dec_lr_avg_loss']) and not dec_lr:
            dec_lr = True
            print('Decrease learning rate to %f' % config['dec_lr'], flush=True)
            del optimizer
            optimizer = get_optimizer(net.parameters(), config['optim'], config['dec_lr'])


if __name__ == '__main__':
    # args: [pretrain]
    if len(sys.argv) > 1:
        if 'pretrain' in sys.argv[1:]:
            config['use_pretrained_model'] = True

    train()
