import argparse
import torch
import sys
import time
import os
import logging
import yaml
import shutil
import numpy as np
import tensorboardX
import torch.optim as optim
import torchvision
from image_transformer import ImageTransformer
import matplotlib
import itertools
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchviz import make_dot
from tqdm import tqdm
import torch.nn as nn

matplotlib.use('Agg')
import seaborn as sns
import matplotlib.pyplot as plt

sns.set()

def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

def parse_args_and_config():
    """
    :return args, config: namespace objects that stores information in args and config files.
    """
    parser = argparse.ArgumentParser(description=globals()['__doc__'])

    parser.add_argument('--config', type=str, default='transformer_dmol.yml', help='Path to the config file')
    parser.add_argument('--doc', type=str, default='0', help='A string for documentation purpose')
    parser.add_argument('--verbose', type=str, default='info', help='Verbose level: info | debug | warning | critical')
    parser.add_argument('--sample', action='store_true', help='Sample at train time')

    args = parser.parse_args()
    args.log = os.path.join('transformer_logs', args.doc)
    # parse config file
    with open(os.path.join('configs', args.config), 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    new_config = dict2namespace({**config, **vars(args)})

    if os.path.exists(args.log):
        shutil.rmtree(args.log)

    os.makedirs(args.log)

    with open(os.path.join(args.log, 'config.yml'), 'w') as f:
        yaml.dump(new_config, f, default_flow_style=False)

    # setup logger
    level = getattr(logging, args.verbose.upper(), None)
    if not isinstance(level, int):
        raise ValueError('level {} not supported'.format(args.verbose))

    handler1 = logging.StreamHandler()
    handler2 = logging.FileHandler(os.path.join(args.log, 'stdout.txt'))
    formatter = logging.Formatter('%(levelname)s - %(filename)s - %(asctime)s - %(message)s')
    handler1.setFormatter(formatter)
    handler2.setFormatter(formatter)
    logger = logging.getLogger()
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    logger.setLevel(level)

    # add device information to args
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info("Using device: {}".format(device))
    new_config.device = device

    # set random seed
    torch.manual_seed(new_config.seed)
    torch.cuda.manual_seed_all(new_config.seed)
    np.random.seed(new_config.seed)
    logging.info("Run name: {}".format(args.doc))

    return args, new_config

def get_lr(step, config):
    warmup_steps = config.optim.warmup
    lr_base = config.optim.lr * 0.002 # for Adam correction
    ret = 5000. * config.model.hidden_size ** (-0.5) * \
          np.min([(step + 1) * warmup_steps ** (-1.5), (step + 1) ** (-0.5)])
    return ret * lr_base

def main():
    args, config = parse_args_and_config()
    tb_logger = tensorboardX.SummaryWriter(log_dir=os.path.join('transformer_logs', args.doc))

    if config.model.distr == "dmol":
        # Scale size and rescale data to [-1, 1]
        transform = transforms.Compose([
            transforms.Resize(config.model.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                 std=[0.5, 0.5, 0.5])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(config.model.image_size),
            transforms.ToTensor()
        ])
    dataset = datasets.CIFAR10('datasets/transformer', transform=transform, download=True)
    loader = DataLoader(dataset, batch_size=config.train.batch_size, shuffle=True, num_workers=4)
    input_dim = config.model.image_size ** 2 * config.model.channels
    model = ImageTransformer(config.model).to(config.device)
    optimizer = optim.Adam(model.parameters(), lr=1., betas=(0.9, 0.98), eps=1e-9)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda step: get_lr(step, config))

    # Initialize as in their code
    gain = config.model.initializer_gain
    for name, p in model.named_parameters():
        if "layernorm" in name:
            continue
        # This is from a pytorch implementation of the language transformer, but is not needed/in TF code.
        # if "attn" in name and "output" not in name:
        #     nn.init.xavier_normal_(p)
        if p.dim() > 1:
            nn.init.xavier_uniform_(p, gain=np.sqrt(gain)) # Need sqrt for inconsistency between pytorch / TF
        else:
            a =  np.sqrt(3. * gain / p.shape[0])
            nn.init.uniform_(p, -a, a)

    # Accumulate data statistics for debugging purposes, e.g. to analyze the entropy of the first dimension
    # data_avgs = torch.zeros(config.model.channels, config.model.image_size, config.model.image_size, 256)
    # for i, (imgs, l) in tqdm(enumerate(loader)):
    #     one_hot_data = torch.zeros(imgs.shape + (256,)).scatter_(-1, (imgs * 255).long().unsqueeze(-1), 1)
    #     data_avgs += one_hot_data.mean(0)
    # data_avgs /= i

    def revert_samples(input):
        if config.model.distr == "cat":
            return input
        elif config.model.distr == "dmol":
            return input * 0.5 + 0.5

    step = 0
    losses_per_dim = torch.zeros(config.model.channels, config.model.image_size, config.model.image_size).to(config.device)
    for _ in range(config.train.epochs):
        for _, (imgs, l) in enumerate(loader):
            imgs = imgs.to(config.device)
            model.train()

            scheduler.step()
            optimizer.zero_grad()
            preds = model(imgs)
            loss = model.loss(preds, imgs)
            decay = 0. if step == 0 else 0.99
            if config.model.distr == "dmol":
                losses_per_dim[0,:,:] = losses_per_dim[0,:,:] * decay + (1 - decay) * loss.detach().mean(0) / np.log(2)
            else:
                losses_per_dim = losses_per_dim * decay + (1 - decay) * loss.detach().mean(0) / np.log(2)
            loss = loss.view(loss.shape[0], -1).sum(1)
            loss = loss.mean(0)

            # Show computational graph
            # dot = make_dot(loss, dict(model.named_parameters()))
            # dot.render('test.gv', view=True)

            loss.backward()

            total_norm = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
            total_norm = (total_norm ** (1. / 2))

            if config.train.clip_grad_norm > 0.0:
                nn.utils.clip_grad_norm_(model.parameters(), config.train.clip_grad_norm)

            total_norm_post = 0
            for p in model.parameters():
                param_norm = p.grad.data.norm(2)
                total_norm_post += param_norm.item() ** 2
            total_norm_post = (total_norm_post ** (1. / 2))

            optimizer.step()
            bits_per_dim = loss / (np.log(2.) * input_dim)
            acc = model.accuracy(preds, imgs)

            if step % config.train.log_iter == 0:
                logging.info('step: {}; loss: {:.3f}; bits_per_dim: {:.3f}, acc: {:.3f}, grad norm pre: {:.3f}, post: {:.3f}'
                             .format(step, loss.item(), bits_per_dim.item(), acc.item(), total_norm, total_norm_post))
                tb_logger.add_scalar('loss', loss.item(), global_step=step)
                tb_logger.add_scalar('bits_per_dim', bits_per_dim.item(), global_step=step)
                tb_logger.add_scalar('acc', acc.item(), global_step=step)
                tb_logger.add_scalar('grad_norm', total_norm, global_step=step)

            if step % config.train.sample_iter == 0:
                logging.info("Sampling from model: {}".format(args.doc))
                if config.model.distr == "cat":
                    channels = ['r','g','b']
                    color_codes = ['Reds', "Greens", 'Blues']
                    for idx, c in enumerate(channels):
                        ax = sns.heatmap(losses_per_dim[idx,:,:].cpu().numpy(), linewidth=0.5, cmap=color_codes[idx])
                        tb_logger.add_figure("losses_per_dim/{}".format(c), ax.get_figure(), close=True, global_step=step)
                else:
                    ax = sns.heatmap(losses_per_dim[0,:,:].cpu().numpy(), linewidth=0.5, cmap='Blues')
                    tb_logger.add_figure("losses_per_dim", ax.get_figure(), close=True, global_step=step)

                model.eval()
                with torch.no_grad():
                    imgs = revert_samples(imgs)
                    imgs_grid = torchvision.utils.make_grid(imgs[:8, ...], 3)
                    tb_logger.add_image('imgs', imgs_grid, global_step=step)

                    # Evaluate model predictions for the input
                    pred_samples = revert_samples(model.sample_from_preds(preds))
                    pred_samples_grid = torchvision.utils.make_grid(pred_samples[:8, ...], 3)
                    tb_logger.add_image('pred_samples/random', pred_samples_grid, global_step=step)
                    pred_samples = revert_samples(model.sample_from_preds(preds, argmax=True))
                    pred_samples_grid = torchvision.utils.make_grid(pred_samples[:8, ...], 3)
                    tb_logger.add_image('pred_samples/argmax', pred_samples_grid, global_step=step)

                    if args.sample:
                        samples = revert_samples(model.sample(config.train.sample_size, config.device))
                        samples_grid = torchvision.utils.make_grid(samples[:8, ...], 3)
                        tb_logger.add_image('samples', samples_grid, global_step=step)

                    # Argmax samples are not useful for unconditional generation
                    # if config.model.distr == "cat":
                    #     argmax_samples = model.sample(1, config.device, argmax=True)
                    #     samples_grid = torchvision.utils.make_grid(argmax_samples[:8, ...], 3)
                    #     tb_logger.add_image('argmax_samples', samples_grid, global_step=step)
                torch.save(model.state_dict(), os.path.join('transformer_logs', args.doc, "model.pth"))
            step += 1

    return 0

if __name__ == '__main__':
    sys.exit(main())
