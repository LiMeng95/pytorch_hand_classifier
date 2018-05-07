from __future__ import print_function

import os
import numpy as np

import torch as t
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.autograd import Variable
from torchnet import meter

from .log import logger
from .visualize import Visualizer


def get_learning_rates(optimizer):
    lrs = [pg['lr'] for pg in optimizer.param_groups]
    lrs = np.asarray(lrs, dtype=np.float)
    return lrs


class TrainParams(object):
    # required params
    max_epoch = 30

    # optimizer and criterion and learning rate scheduler
    optimizer = None
    criterion = None
    lr_scheduler = None         # should be an instance of ReduceLROnPlateau or _LRScheduler

    # params based on your local env
    use_gpu = False             # default do not use gpu
    save_dir = './models/'            # default `save_dir`

    # loading existing checkpoint
    ckpt = None                 # path to the ckpt file

    # saving checkpoints
    save_freq_epoch = 1         # save one ckpt per `save_freq_epoch` epochs


class Trainer(object):

    TrainParams = TrainParams

    def __init__(self, model, train_params, train_data, val_data=None):
        assert isinstance(train_params, TrainParams)
        self.params = train_params

        # Data loaders
        self.train_data = train_data
        self.val_data = val_data

        # criterion and Optimizer and learning rate
        self.last_epoch = 0
        self.criterion = self.params.criterion
        self.optimizer = self.params.optimizer
        self.lr_scheduler = self.params.lr_scheduler
        logger.info('Set criterion to {}'.format(type(self.criterion)))
        logger.info('Set optimizer to {}'.format(type(self.optimizer)))
        logger.info('Set lr_scheduler to {}'.format(type(self.lr_scheduler)))

        # load model
        self.model = model
        logger.info('Set output dir to {}'.format(self.params.save_dir))
        if os.path.isdir(self.params.save_dir):
            pass
        else:
            os.makedirs(self.params.save_dir)

        ckpt = self.params.ckpt
        if ckpt is not None:
            self._load_ckpt(ckpt)
            logger.info('Load ckpt from {}'.format(ckpt))

        # meters
        self.loss_meter = meter.AverageValueMeter()
        self.confusion_matrix = meter.ConfusionMeter(6)

        # set CUDA_VISIBLE_DEVICES
        if self.params.use_gpu:
            logger.info('Set CUDA_VISIBLE_DEVICES to 0...')
            self.model = self.model.cuda()

        self.model.train()

    def train(self):
        vis = Visualizer()
        best_loss = np.inf
        for epoch in range(self.last_epoch, self.params.max_epoch):

            self.loss_meter.reset()
            self.confusion_matrix.reset()

            self.last_epoch += 1
            logger.info('Start training epoch {}'.format(self.last_epoch))

            self._train_one_epoch()

            # save model
            if (self.last_epoch % self.params.save_freq_epoch == 0) or (self.last_epoch == self.params.max_epoch - 1):
                save_name = self.params.save_dir + 'ckpt_epoch_{}.pth'.format(self.last_epoch)
                t.save(self.model.state_dict(), save_name)

            val_cm, val_accuracy = self._val_one_epoch()

            if self.loss_meter.value()[0] < best_loss:
                logger.info('Found a better ckpt ({:.3f} -> {:.3f}), '.format(best_loss, self.loss_meter.value()[0]))
                best_loss = self.loss_meter.value()[0]

            # visualize
            vis.plot('loss', self.loss_meter.value()[0])
            vis.plot('val_accuracy', val_accuracy)
            vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
                epoch=epoch, loss=self.loss_meter.value()[0], val_cm=str(val_cm.value()),
                train_cm=str(self.confusion_matrix.value()), lr=get_learning_rates(self.optimizer)))

            # adjust the lr
            if isinstance(self.lr_scheduler, ReduceLROnPlateau):
                self.lr_scheduler.step(self.loss_meter.value()[0], self.last_epoch)

    def _load_ckpt(self, ckpt):
        self.model.load_state_dict(t.load(ckpt))

    def _train_one_epoch(self):
        for step, (data, label) in enumerate(self.train_data):
            # train model
            inputs = Variable(data)
            target = Variable(label)
            if self.params.use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            # forward
            score = self.model(inputs)
            loss = self.criterion(score, target)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step(None)

            # meters update
            self.loss_meter.add(loss.data[0])
            self.confusion_matrix.add(score.data, target.data)

    def _val_one_epoch(self):
        self.model.eval()
        confusion_matrix = meter.ConfusionMeter(6)
        logger.info('Val on validation set...')

        for step, (data, label) in enumerate(self.val_data):

            # val model
            inputs = Variable(data, volatile=True)
            target = Variable(label.type(t.LongTensor), volatile=True)
            if self.params.use_gpu:
                inputs = inputs.cuda()
                target = target.cuda()

            score = self.model(inputs)
            confusion_matrix.add(score.data.squeeze(), label.type(t.LongTensor))

        self.model.train()
        cm_value = confusion_matrix.value()
        accuracy = 100. * (cm_value[0][0] + cm_value[1][1]
                           + cm_value[2][2] + cm_value[3][3]
                           + cm_value[4][4] + cm_value[5][5]) / (cm_value.sum())
        return confusion_matrix, accuracy
