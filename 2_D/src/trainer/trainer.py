import numpy as np
import torch
from torchvision.utils import make_grid

from src.trainer.base import BaseTrainer
from src.utils.utils import inf_loop, MetricTracker


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = config['trainer']['log_step']

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, batch_sample in enumerate(self.data_loader):
            batch_sample = {k: v.to(self.device) for k,v in batch_sample.items()}

            self.optimizer.zero_grad()
            input = (batch_sample['frame0'], batch_sample['frame2'], batch_sample['frame4'])
            output = self.model(*input)
            target = (batch_sample['frame1'], batch_sample['frame2'], batch_sample['frame3'])
            loss = self.criterion(output, target, **self.config['loss']['args'])
            loss.backward()
            self.optimizer.step()

            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(output, target))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.6f}'.format(
                    epoch,
                    self._progress(batch_idx),
                    loss.item()))
                self._show_images(input, target, output)

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_'+k : v for k, v in val_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, batch_sample in enumerate(self.valid_data_loader):
                batch_sample = {k: v.to(self.device) for k,v in batch_sample.items()}

                input = (batch_sample['frame0'], batch_sample['frame2'], batch_sample['frame4'])
                output = self.model(*input)
                target = (batch_sample['frame1'], batch_sample['frame2'], batch_sample['frame3'])
                loss = self.criterion(output, target, **self.config['loss']['args'])

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(output, target))
                self._show_images(input, target, output)

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    def _show_images(self, input, target, output):
        b, c, w, h = input[0].shape
        for i in range(b):
            images = torch.cat([
                torch.cat([input[0][i], input[1][i], input[2][i]], dim=0).view(3, c, w, h),
                torch.cat([target[0][i], target[1][i], target[2][i]], dim=0).view(3, c, w, h),
                torch.cat([output[0][i], output[1][i], output[2][i]], dim=0).view(3, c, w, h),
            ], dim=0)
            self.writer.add_image(f'example_{i}', make_grid(images.cpu(), nrow=3, normalize=False))
