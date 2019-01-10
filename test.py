import data_gen
import models
import util

import numpy as np
import pandas as pd
import heapq

import torch
from torch.autograd.variable import Variable
import os
import re
import sys

data = None
words = None
ed = None


config = util.config_test


def mock(doc_w, doc_e, doc_ed):
    return np.random.rand(doc_e.shape[0], 1)


def testData(value: pd.DataFrame):
    global data, words, ed
    L = 10
    if '' in value.pos_entity:
        value.pos_entity.remove('')
    if '' in value.neg_entity:
        value.neg_entity.remove('')

    pos_e_shape = (len(value.pos_entity), 128)
    neg_e_shape = (len(value.neg_entity), 128)

    if value.pos_entity == ['']:
        pos_e_shape = (0, 128)
    if value.neg_entity == ['']:
        neg_e_shape = (0, 128)

    pos_ed_shape = (pos_e_shape[0], L, 128)
    neg_ed_shape = (neg_e_shape[0], L, 128)

    pos_e = np.ones(pos_e_shape, dtype=np.float)
    for i in range(pos_e_shape[0]):
        pos_e[i] = words.loc[value.pos_entity[i]]

    pos_ed = np.ones(pos_ed_shape, dtype=np.float)
    for i in range(pos_ed_shape[0]):
        desciption = ed.loc[value.pos_entity[i]]
        desciption = list(map(lambda x: x if x is not '' else '</s>', desciption))
        for j in range(pos_ed_shape[1]):
            pos_ed[i][j] = words.loc[desciption[j].lower()]

    neg_e = np.ones(neg_e_shape, dtype=np.float)
    for i in range(neg_e_shape[0]):
        neg_e[i] = words.loc[value.neg_entity[i]]

    neg_ed = np.ones(neg_ed_shape, dtype=np.float)
    for i in range(neg_ed_shape[0]):
        desciption = ed.loc[value.neg_entity[i]]
        desciption = list(map(lambda x: x if x is not '' else '</s>', desciption))
        for j in range(neg_ed_shape[1]):
            neg_ed[i][j] = words.loc[desciption[j].lower()]

    abstract_words = value.paperAbstract.replace(',', '').lower()
    abstract_words = abstract_words.replace('.', '')
    abstract_words = abstract_words.strip()
    abstract_words = abstract_words.replace('\n', ' ')  # re.split(' |\n', abstract_words)
    abstract_words = abstract_words.split(' ')
    abstract_words = list(map(lambda x: x if x is not '' else '</s>', abstract_words))
    doc_w_shape = (len(abstract_words), 128)
    doc_w = np.ones(doc_w_shape, dtype=np.float)
    for i in range(doc_w_shape[0]):
        try:
            doc_w[i] = words.loc[abstract_words[i]]
        except KeyError:
            doc_w[i] = words.loc['</s>']

    doc_e_shape = (neg_e_shape[0] + pos_e_shape[0], 128)
    doc_e = np.concatenate([pos_e, neg_e])

    doc_ed_shape = (neg_e_shape[0] + pos_e_shape[0], L, 128)
    doc_ed = np.concatenate([pos_ed, neg_ed])

    return doc_e.astype(np.float32), doc_ed.astype(np.float32), doc_w.astype(np.float32)


def test(net):
    global data, words, ed
    data = data_gen.data
    data['all_entity'] = data[['neg_entity', 'pos_entity']].apply(lambda x : np.concatenate([x.pos_entity, x.neg_entity]), axis = 1)
    data['pc'] = data['pos_entity'].map(lambda x : len(x))
    data = data[data.pc != 0]
    data['nc'] = data['neg_entity'].map(lambda x: len(x))
    data = data[data.nc != 0]
    words = data_gen.words
    ed = data_gen.entity_description

    i = 0
    total = len(data)
    precision_1 = np.zeros(total, dtype=np.float32)
    precision_5 = np.zeros(total, dtype=np.float32)
    recall_1 = np.zeros(total, dtype=np.float32)
    recall_5 = np.zeros(total, dtype=np.float32)

    for _, value in data.iterrows():
        # doc_E, doc_desp, doc_w
        np_inputs = testData(value)
        torch_inputs = tuple(Variable(torch.from_numpy(arr)) for arr in np_inputs)
        if config['gpu']:
            torch_inputs = tuple(arr.cuda() for arr in torch_inputs)

        scores = net.score(torch_inputs).cpu().numpy()

        if len(value.all_entity) == 0:
            precision_1[i] = -1.0
            precision_5[i] = -1.0
            recall_1[i] = -1.0
            recall_5[i] = -1.0
            i += 1
            continue

        if len(value.pos_entity) == 0:
            precision_1[i] = 0.0
            precision_5[i] = 0.0
            recall_1[i] = 1.0
            recall_5[i] = 1.0
            i += 1
            continue

        if value.all_entity[scores.argmax()] in value.pos_entity:
            precision_1[i] = 1.0
            recall_1[i] = 1.0 / float(len(value.pos_entity)) if len(value.pos_entity) != 0 else 1.0
        else:
            precision_1[i] = 0.0
            recall_1[i] = 0.0

        top_5 = heapq.nlargest(5, range(len(scores)), scores.take)
        right = 0
        for pos in top_5:
            if value.all_entity[pos] in value.pos_entity:
                right += 1

        precision_5[i] = right / 5.0
        recall_5[i] = right / float(len(value.pos_entity)) if len(value.pos_entity) != 0 else 1.0

        i += 1

    return precision_1, precision_5, recall_1, recall_5


def eval_model(epochs):
    net = models.KESMSalienceEstimation(config['esize'], config['wsize'], None, config['keesize'], config['L'],
                                 config['cnn_ksize'], config['K'], config['cos_epsilon'])
    data_gen.prepare('data_test/articles_testset.csv', 'data_test/entities_testset.csv',
                     'data_test/reformat_testset.json', 'data_all/maps.txt', 'data_all/w2v_vec')

    print('Testing model %s' % config['model_prefix'])
    for epoch in epochs:
        model_name = '%s_%d.pth' % (config['model_prefix'], epoch)
        net.load_state_dict(torch.load(config['model_dir'] + '/' + model_name))
        if config['gpu']:
            net.cuda()
        for param in net.parameters():
            param.requires_grad = False
        net.eval()

        p1, p5, r1, r5 = (np.mean(x) for x in test(net))

        print('Epoch %d: P@1 = %.4f, P@5 = %.4f, R@1 = %.4f, R@5 = %.4f' % (epoch, p1, p5, r1, r5))


if __name__ == '__main__':
    if len(sys.argv) > 1:
        test_epochs = [int(ep) for ep in sys.argv[1:]]
        eval_model(test_epochs)
    else:
        raise ValueError('No model epoch specified to test')
