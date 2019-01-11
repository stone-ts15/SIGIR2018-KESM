
import os
import re


config_train = {
    'esize': 128,
    'wsize': 128,
    'keesize': 128,
    'L': 10,
    'cnn_ksize': 3,
    'K': 11,
    'cos_epsilon': 1e-6,
    'dropout_p': 0.4,

    'gpu': True,
    'optim': 'Adam',
    'weight_decay': 0.0,
    'lr': 1e-4,
    'dec_lr': 1e-5,
    'epochs': 100,
    'dec_lr_epoch': 20,
    'dec_lr_avg_loss': 0.02,
    'model_dir': 'trained_models',
    'model_prefix': 'trainset_bn_tanh_adam_batch128_drop',
    'batch_size': 128,
    'use_pretrained_model': False
}

config_test = {
    'esize': 128,
    'wsize': 128,
    'keesize': 128,
    'L': 10,
    'cnn_ksize': 3,
    'K': 11,
    'cos_epsilon': 1e-6,
    'dropout_p': 0.4,

    'gpu': True,
    'model_dir': 'trained_models',
    'model_prefix': 'trainset_bn_tanh_adam_batch128_drop'
}


def last_model(model_dir):
    last_epoch = -1
    last_model_name = None
    for model_name in os.listdir(model_dir):
        try:
            epoch = re.search(r'^' + config_train['model_prefix'] + r'_(\d+)', model_name).group(1)
            epoch = int(epoch)
            if epoch > last_epoch:
                last_epoch = epoch
                last_model_name = model_name
        except:
            pass

    if last_model_name is not None:
        return last_epoch, last_model_name
    else:
        raise ValueError('No valid model name in %s' % model_dir)