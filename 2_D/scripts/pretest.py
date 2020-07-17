import os
import argparse
from datetime import datetime

from PIL import Image
import torch
import numpy as np

import src.data_loader.data_loaders as module_data
import src.model.loss as module_loss
import src.model.metric as module_metric
import src.model.model as module_arch
from src.utils.parse_config import ConfigParser


def main(config):
    logger = config.get_logger('test')

    # setup data_loader instance
    data_loader = config.init_obj('pretest_data_loader', module_data)

    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    loss_fn = getattr(module_loss, "l1_loss")
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_loss = 0.0
    total_metrics = torch.zeros(len(metric_fns))

    exper_name = config['name']
    run_id = datetime.now().strftime(r'%m%d_%H%M%S')
    save_folder = os.path.join(config['trainer']['save_dir'], 'test', exper_name, run_id)

    os.makedirs(save_folder)

    with torch.no_grad():
        for batch_idx, batch_sample in enumerate(data_loader):
            for k, v in batch_sample.items():
                if '_path' not in k:
                    batch_sample[k] = v.to(device)
            output = model.forward_single(batch_sample['frame0'], batch_sample['frame2'])
            target = batch_sample['frame1']
            batch_size = batch_sample['frame0'].shape[0]

            frames1 = output.cpu().detach().numpy()
            true_frames1 = batch_sample['frame0'].cpu().detach().numpy()
            true_frames2 = batch_sample['frame1'].cpu().detach().numpy()
            true_frames3 = batch_sample['frame2'].cpu().detach().numpy()

            for i in range(batch_size):
                frame1 = frames1[i].astype(np.float32)
                true_frame1 = true_frames1[i].astype(np.float32)
                true_frame2 = true_frames2[i].astype(np.float32)
                true_frame3 = true_frames3[i].astype(np.float32)
                frame1 = np.moveaxis(frame1, 0, 2)
                true_frame1 = np.moveaxis(true_frame1, 0, 2)
                true_frame2 = np.moveaxis(true_frame2, 0, 2)
                true_frame3 = np.moveaxis(true_frame3, 0, 2)
                frame1 = (np.clip(frame1, 0, 1) * 255).astype(np.uint8)
                true_frame1 = (np.clip(true_frame1, 0, 1) * 255).astype(np.uint8)
                true_frame2 = (np.clip(true_frame2, 0, 1) * 255).astype(np.uint8)
                true_frame3 = (np.clip(true_frame3, 0, 1) * 255).astype(np.uint8)
                frame1_path = batch_sample['frame0_path'][i]
                frame2_path = batch_sample['frame1_path'][i]
                frame3_path = batch_sample['frame2_path'][i]
                Image.fromarray(true_frame1).save(os.path.join(save_folder, frame1_path))
                Image.fromarray(true_frame2).save(os.path.join(save_folder, frame2_path))
                Image.fromarray(true_frame3).save(os.path.join(save_folder, frame3_path))
                Image.fromarray(frame1).save(os.path.join(save_folder, frame1_path + 'pred.png'))

                # computing loss, metrics on test set
                loss = loss_fn(output, target)
                total_loss += loss.item() * batch_size
                for i, metric in enumerate(metric_fns):
                    total_metrics[i] += metric(output, target) * batch_size

        n_samples = len(data_loader.sampler)
        log = {'loss': total_loss / n_samples}
        log.update({
            met.__name__: total_metrics[i].item() / n_samples for i, met in enumerate(metric_fns)
        })
        logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='Cartoon interpolation test')
    args.add_argument('-c', '--config', default=None, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default=None, type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
