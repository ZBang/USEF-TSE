# -*- coding: utf-8 -*-

import os
import time
import torch
import torch.nn as nn
import numpy as np

from utils.losses import batchMean_sisnrLoss

class Trainer(object):
    
    def __init__(self, chkpt_dir, data, model, optimizer, scheduler, logger, config):
        self.tr_loader = data['tr_loader']
        self.cv_loader = data['cv_loader']
        
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        # Training config
        self.epochs = config['epochs']
        self.max_norm = config['max_norm']
        self.logger = logger
        # save and load model
        self.save_folder = chkpt_dir
        self.checkpoint = config['checkpoint']
        self.continue_from = config['continue_from']
        # logging
        self.print_freq = config['print_freq']
        os.makedirs(self.save_folder, exist_ok=True)
        self.best_val_loss = float("inf")
        self.start_epoch = 0
        
        if self.continue_from:
            print('Loading checkpoint model %s' % self.continue_from)
            cont = torch.load(self.continue_from)
            self.start_epoch = cont['epoch']
            self.model.load_state_dict(cont['model_state_dict'])
            self.optimizer.load_state_dict(cont['optimizer_state'])
            torch.set_rng_state((cont['trandom_state']))
            np.random.set_state((cont['nrandom_state']))

    def _run_train_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        data_loader = self.tr_loader

        for i, (data) in enumerate(data_loader):

            mixture, source, embd, ilens = data
            mixture = mixture.cuda()
            source = source.cuda()
            embd = embd.cuda()
            ilens = ilens.cuda()
            
            estimate_source = self.model(mixture, embd)

            loss = batchMean_sisnrLoss(estimate_source, source)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                            self.max_norm)
            self.optimizer.step()

            total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Loss {2:3.3f} | '
                    'Current Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)))

        torch.cuda.empty_cache()

        return total_loss / (i + 1)
    
    def _run_valid_epoch(self, epoch):

        start = time.time()
        total_loss = 0
        data_loader = self.cv_loader

        for i, (data) in enumerate(data_loader):

            mixture, source, embd, ilens = data
            mixture = mixture.cuda()
            source = source.cuda()
            embd = embd.cuda()
            ilens = ilens.cuda()
            
            estimate_source = self.model(mixture, embd)
            min_len = min(estimate_source.shape[1], source.shape[1])
            loss = batchMean_sisnrLoss(estimate_source[:,:min_len], source[:,:min_len])

            total_loss += loss.item()

            if i % self.print_freq == 0:
                self.logger.info('Epoch {0:3d} | Iter {1:5d} | Average Valid Loss {2:3.3f} | '
                    'Current Valid Loss {3:3.6f} | {4:5.1f} ms/batch'.format(
                        epoch + 1, i + 1, total_loss / (i + 1),
                        loss.item(), 1000 * (time.time() - start) / (i + 1)))
        
        torch.cuda.empty_cache()

        return total_loss / (i + 1)
    
    def train(self):
        # Train model multi-epoches
        for epoch in range(self.start_epoch, self.epochs):

            optim_state = self.optimizer.state_dict()
            self.logger.info('epoch start Learning rate: {lr:.6f}'.format(lr=optim_state['param_groups'][0]['lr']))
            self.logger.info("Training...")
            
            # train stage
            self.model.train()
            start = time.time()
            tr_loss = self._run_train_epoch(epoch)

            # train log
            self.logger.info('-' * 85)
            self.logger.info('Train Summary | End of Epoch {0:5d} | Time {1:.2f}s | '
                'Train Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, tr_loss))
            self.logger.info('-' * 85)

            # Save model each epoch
            if self.checkpoint:
                file_path = os.path.join(
                    self.save_folder, 'epoch%d.pth.tar' % (epoch + 1))
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, file_path)
                self.logger.info('Saving checkpoint model to %s' % file_path)

            # validation stage
            self.logger.info('Cross validation...')
            
            self.model.eval()
            with torch.no_grad():
                val_loss = self._run_valid_epoch(epoch)
                
            # val log
            self.logger.info('-' * 85)
            self.logger.info('Valid Summary | End of Epoch {0} | Time {1:.2f}s | '
                'Valid Loss {2:.3f}'.format(
                    epoch + 1, time.time() - start, val_loss))
            self.logger.info('-' * 85)

            # save the temp_best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_file_path = os.path.join(
                    self.save_folder, 'temp_best.pth.tar')
                torch.save({
                    'epoch': epoch+1,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'trandom_state': torch.get_rng_state(),
                    'nrandom_state': np.random.get_state()}, best_file_path)
                self.logger.info("Find better validated model, saving to %s" % best_file_path)
            
            self.scheduler.step(val_loss) 
