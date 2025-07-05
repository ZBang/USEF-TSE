import os
import sys
import torch
sys.path.append('../../..')
import numpy as np
import logging
import argparse
import time
import random
import shutil
import inspect

from hyperpyyaml import load_hyperpyyaml
from torch.utils.data.distributed import DistributedSampler
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.distributed as dist

from trainer.trainer import Trainer
# from utils.freeze import *

from dataset.data import tr_dataset, te_dataset

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.cuda.manual_seed_all(SEED)


def main(config, args):

    log_dir = os.path.join('logs', config['name'])
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(
        log_dir, time.strftime('%Y-%m-%d-%H%M.log',time.localtime(time.time()))
    )
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger()

    trainset = tr_dataset(
        mix_scp = os.path.join(config['train_path'], config['mix_scp']),
        ref_scp = os.path.join(config['train_path'], config['ref_scp']),
        aux_scp = os.path.join(config['train_path'], config['aux_scp']),
        dur = config['duration'],
        fs = config['sample_rate']
    )

    trainloader = DataLoader(
            dataset=trainset, 
            batch_size=config['batch_size'],
            shuffle=True,
            num_workers=config['num_workers'],
            pin_memory=True,
            sampler=None,
    )
    
    validset = te_dataset(
        mix_scp = os.path.join(config['valid_path'], config['mix_scp']),
        ref_scp = os.path.join(config['valid_path'], config['ref_scp']),
        aux_scp = os.path.join(config['valid_path'], config['aux_scp']),
        fs = config['sample_rate']
    )

    validloader = DataLoader(
            dataset=validset, 
            batch_size=config['valid_batch_size'],
            shuffle=False,
            num_workers=config['num_workers'],
            pin_memory=True,
            sampler=None,
    )
    
    data = {'tr_loader': trainloader, 'cv_loader': validloader}

    model = config['modules']['masknet']
    
    model.cuda()
    
    logger.info(model)
    logger.info('-' * 50)

    model = torch.nn.DataParallel(model)

    chkpt_dir = os.path.join('chkpt', config['name'])
    os.makedirs(chkpt_dir, exist_ok=True)

    shutil.copyfile(args.config, os.path.join(chkpt_dir, 'config.yaml'))
    shutil.copyfile(inspect.getmodule(config['MaskNet'].__class__).__file__, os.path.join(chkpt_dir, 'model.py'))

    model_params = list(filter(lambda p: p.requires_grad, model.parameters()))
    optimizer = config['optimizer'](params=model_params)
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

    trainer = Trainer(chkpt_dir = chkpt_dir,
                      data = data,
                      model = model,
                      optimizer = optimizer,
                      scheduler= lr_scheduler,
                      logger = logger,
                      config = config,
                      
    )
    trainer.train()


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Speech Separation: transformer')
    parser.add_argument('--config', default='config/config.yaml', type=str,
                        help='config file path (default: None)')
    
    args = parser.parse_args()

    # Read config of the whole system.
    assert os.path.isfile(args.config), "No such file: %s" % args.config
    with open(args.config, 'r') as f:
        config_strings = f.read()
    config = load_hyperpyyaml(config_strings)
    print('INFO: Loaded hparams from: {}'.format(args.config))

    main(config, args)
