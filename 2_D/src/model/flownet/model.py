from os.path import *
import argparse
import colorama
import os
import subprocess
import sys
import setproctitle

sys.path.append('submodules/flownet2-pytorch')
sys.path.append('../../../submodules/flownet2-pytorch')

import cv2
import cvbase as cvb
import torch
import torch.nn as nn

from utils import flow_utils, tools
import datasets
import models
import losses


flow_net = None


def _get_flow_net():
    parser = argparse.ArgumentParser()

    parser.add_argument('--batch_size', '-b', type=int, default=8, help="Batch size")
    parser.add_argument('--train_n_batches', type=int, default=-1,
                        help='Number of min-batches per epoch. If < 0, it will be determined by training_dataloader')
    parser.add_argument('--crop_size', type=int, nargs='+', default=[256, 256],
                        help="Spatial dimension to crop training samples for training")
    parser.add_argument('--gradient_clip', type=float, default=None)
    parser.add_argument('--schedule_lr_frequency', type=int, default=0,
                        help='in number of iterations (0 for no schedule)')
    parser.add_argument('--schedule_lr_fraction', type=float, default=10)
    parser.add_argument("--rgb_max", type=float, default=255.)
    parser.add_argument('--number_workers', '-nw', '--num_workers', type=int, default=8)
    parser.add_argument('--number_gpus', '-ng', type=int, default=-1, help='number of GPUs to use')
    parser.add_argument('--no_cuda', action='store_true')
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--name', default='run', type=str, help='a name to append to the save directory')
    parser.add_argument('--save', '-s', default='./work', type=str, help='directory for saving')
    parser.add_argument('--validation_frequency', type=int, default=5, help='validate every n epochs')
    parser.add_argument('--validation_n_batches', type=int, default=-1)
    parser.add_argument('--render_validation', action='store_true',
                        help='run inference (save flows to file) and every validation_frequency epoch')
    parser.add_argument('--inference', action='store_true')
    parser.add_argument('--inference_size', type=int, nargs='+', default=[-1, -1],
                        help='spatial size divisible by 64. default (-1,-1) -largest possible valid size would be used')
    parser.add_argument('--inference_batch_size', type=int, default=1)
    parser.add_argument('--inference_n_batches', type=int, default=-1)
    parser.add_argument('--resume', default='FlowNet2_checkpoint.pth.tar', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--log_frequency', '--summ_iter', type=int, default=1, help="Log every n batches")
    parser.add_argument('--fp16', action='store_true', help='Run model in pseudo-fp16 mode (fp16 storage fp32 math).')
    parser.add_argument('-c', '--config', default=None, type=str, help='config file path (default: None)')
    parser.add_argument('--image1', type=str, default='', help="Path to the first image")
    parser.add_argument('--image2', type=str, default='', help="Path to the second image")

    parser.add_argument('-r', '--resume-old', type=str)
    tools.add_arguments_for_module(parser, models, argument_for_class='model', default='FlowNet2')
    tools.add_arguments_for_module(parser, losses, argument_for_class='loss', default='L1Loss')
    tools.add_arguments_for_module(parser, torch.optim, argument_for_class='optimizer', default='Adam',
                                   skip_params=['params'])
    tools.add_arguments_for_module(parser, datasets, argument_for_class='training_dataset', default='MpiSintelFinal',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel-complete/training/'})
    tools.add_arguments_for_module(parser, datasets, argument_for_class='validation_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel-complete/training/', 'replicates': 1})
    tools.add_arguments_for_module(parser, datasets, argument_for_class='inference_dataset', default='MpiSintelClean',
                                   skip_params=['is_cropped'],
                                   parameter_defaults={'root': './MPI-Sintel-complete/training/', 'replicates': 1})

    prev_main_dir = os.getcwd()
    main_dir = os.path.dirname(os.path.realpath(__file__))
    os.chdir(main_dir)

    # Parse the official arguments
    with tools.TimerBlock("Parsing Arguments") as block:
        args = parser.parse_args()
        args.number_gpus = 1

        # Get argument defaults (hastag #thisisahack)
        parser.add_argument('--IGNORE', action='store_true')
        defaults = vars(parser.parse_args(['--IGNORE']))

        # Print all arguments, color the non-defaults
        for argument, value in sorted(vars(args).items()):
            reset = colorama.Style.RESET_ALL
            color = reset if value == defaults[argument] else colorama.Fore.MAGENTA
            block.log('{}{}: {}{}'.format(color, argument, value, reset))

        args.model_class = tools.module_to_dict(models)[args.model]
        args.optimizer_class = tools.module_to_dict(torch.optim)[args.optimizer]
        args.loss_class = tools.module_to_dict(losses)[args.loss]

        args.inference_dataset_class = tools.module_to_dict(datasets)[args.inference_dataset]

        args.cuda = not args.no_cuda and torch.cuda.is_available()
        args.current_hash = subprocess.check_output(["git", "rev-parse", "HEAD"]).rstrip()
        args.log_file = join(args.save, 'args.txt')

        # dict to collect activation gradients (for training debug purpose)
        args.grads = {}

    print('Source Code')
    print(('  Current Git Hash: {}\n'.format(args.current_hash)))

    # Change the title for `top` and `pkill` commands
    setproctitle.setproctitle(args.save)

    # Dynamically load model and loss class with parameters passed in via
    # "--model_[param]=[value]" or "--loss_[param]=[value]" arguments
    with tools.TimerBlock("Building {} model".format(args.model)) as block:
        class FlowNet2(nn.Module):
            def __init__(self, args):
                super(FlowNet2, self).__init__()
                kwargs = tools.kwargs_from_args(args, 'model')
                self.model = args.model_class(args, **kwargs)
                kwargs = tools.kwargs_from_args(args, 'loss')
                self.loss = args.loss_class(args, **kwargs)

            def forward(self, data):
                return self.model(data)

        model = FlowNet2(args)

        args.effective_batch_size = args.batch_size * args.number_gpus
        block.log('Effective Batch Size: {}'.format(args.effective_batch_size))
        block.log('Number of parameters: {}'.format(
            sum([p.data.nelement() if p.requires_grad else 0 for p in model.parameters()])))

        # assign to cuda or wrap with data parallel, model and loss
        block.log('Initializing CUDA')
        model = model.cuda()
        block.log('Parallelizing')
        model = nn.parallel.DataParallel(model, device_ids=list(range(args.number_gpus)))
        torch.cuda.manual_seed(args.seed)

        # Load weights if needed, otherwise randomly initialize
        block.log("Loading checkpoint '{}'".format(args.resume))
        checkpoint = torch.load(args.resume)
        model.module.model.load_state_dict(checkpoint['state_dict'])
        block.log("Loaded checkpoint '{}' (at epoch {})".format(args.resume, checkpoint['epoch']))

    os.chdir(prev_main_dir)

    return model


def get_flow_net():
    global flow_net
    if flow_net is None:
        flow_net = _get_flow_net()
    return flow_net


def compute_flow():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image1', type=str, default='', help="Path to the first image")
    parser.add_argument('--image2', type=str, default='', help="Path to the second image")
    args = parser.parse_args()

    image1 = cv2.imread(args.image1)
    image2 = cv2.imread(args.image2)

    image1 = cv2.resize(image1, dsize=(512, 512))
    image2 = cv2.resize(image2, dsize=(512, 512))

    image1 = torch.from_numpy(image1).float()
    image2 = torch.from_numpy(image2).float()

    images = torch.cat([image1[None], image2[None]])
    images = images[None]
    images = images.permute(0, 4, 1, 2, 3)

    flow_net = get_flow_net()
    flow = flow_net(images)
    cvb.show_flow(flow[0].permute(1, 2, 0).cpu().detach().numpy())


if __name__ == '__main__':
    compute_flow()
