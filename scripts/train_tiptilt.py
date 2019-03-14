"""
Trains a Predictive Wavefront Model from scratch
"""

import os
import csv
import math
import time
import git
import sys
import platform
from argparse import ArgumentParser
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ExponentialLR
from tensorboardX import SummaryWriter

# Use Agg rendering on a headless server
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import aopwc


def run_epoch(epoch, dataloader, model, config, tensorboard, optimizer=None):
    
    # Initialize metrics
    total_loss = 0
    total_sqr_error = 0

    # Set model to training or validation mode
    model.train() if optimizer is not None else model.eval()
    phase = 'train' if optimizer is not None else 'val'
    
    # Iterate over examples in the dataset
    for i, data in enumerate(tqdm(dataloader)):


        # Move inputs to GPU
        if config.gpu >= 0:
            data = data.cuda()

        # Run model to generate predicted wavefronts
        predictions = model(data)

        # Compute loss (on normalized dataset)
        loss = F.smooth_l1_loss(predictions, data)
        total_loss += float(loss)

        # Update parameters if we are in training mode
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Compute squared error (on un-normalized dataset)
        sqr_error = F.mse_loss(predictions[:, :, config.steps_ahead:],
                               data[:, :, config.steps_ahead:])
        total_sqr_error += float(sqr_error)

        # Visualize predictions
        if i % config.vis_interval == 0 and tensorboard is not None:
            
            # Plot real and predicted tip-tilt trajectories
            fig = plt.figure('tto')
            fig.clear()
            aopwc.vis_open_loop_tracks(predictions, data, fig.gca())

            # Save to tensorboard
            tensorboard.add_image(phase + '/open-loop-preds',
                aopwc.convert_figure(fig), epoch)


    # Compute average loss and rms
    count = len(dataloader)
    avg_loss = total_loss / count
    rms_error = math.sqrt(total_sqr_error / count)

    return avg_loss, rms_error


def create_experiment(config):
    
    # Print config information
    for key, value in config.__dict__.items():
        print('{}: {}'.format(key, value))
    
    if config.name is None:
        print('\nWARNING: no name specified for experiments: ' \
              'results will not be saved')
        return None, None
    
    # Check that there isn't already an experiment with the given name
    suffix = time.strftime("%Y-%m-%d-%H-%M-%S") + '-' + platform.node()
    logdir = os.path.join(config.logdir, config.name, suffix)
    
    # Create the experiment
    print('\nExperiment directory:\n  {}'.format(logdir))
    os.makedirs(logdir)

    # Save the config data to a YAML file
    aopwc.save_config(logdir, config)
    
    # Create tensorboard summary writer
    print('Creating tensorboard summary...')
    tensorboard = SummaryWriter(log_dir=logdir, comment='')

    # Get current GIT SHA tag
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha

    # Output debug stuff to tensorboard strings
    tensorboard.add_text('command_line', ' '.join(sys.argv), 0)
    tensorboard.add_text('git_sha', str(git_sha), 0)
    tensorboard.add_text('config', str(config.__dict__), 0)

    return logdir, tensorboard


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment. If left blank,'
                             + ' no logging information will be saved to disk')
    parser.add_argument('--logdir', '-d', type=str, 
                        default='experiments',
                        help='location to store experiment log files')
    parser.add_argument('--batch-size', '-b', type=int, default=32,
                        help='number of examples per mini-batch')
    parser.add_argument('--lr', '-l', type=float, default=0.1,
                        help='learning rate')
    parser.add_argument('--momentum', '-m', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--wd', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--steps-ahead', '-s', type=int, default=2,
                        help='number of timesteps into the future to predict')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='index of current gpu (use -1 for cpu training)')
    parser.add_argument('--hidden', type=int, nargs='+',
                        default=[16, 32],
                        help='number of feature channels in hidden layers')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='number of worker threads for data loading'
                             + ' (use 0 for single-threaded mode)')
    parser.add_argument('--vis-interval', type=int, default=10,
                        help='visualize predictions every N iterations')
    parser.add_argument('--gamma', type=float, default=0.95,
                        help='exponential lr decay factor (gamma); use 1.0 for no scheduling')
    return parser.parse_args()


def main():

    # Parse command line arguments
    config = parse_args()

    # Create directory for the experiment and save the config
    logdir, tensorboard = create_experiment(config)

    # Load training an validation datasets
    print('Loading train data...')
    train_data = aopwc.TipTiltDataset('./data/keck_tiptilt/train')
    print('Loading validation data...')
    val_data = aopwc.TipTiltDataset('./data/keck_tiptilt/val')


    # Create dataloaders
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, 
                              num_workers=config.workers)
    val_loader = DataLoader(val_data, config.batch_size, shuffle=True, 
                            num_workers=config.workers)

    # Build model
    model = aopwc.TipTiltNetwork(config.hidden, config.steps_ahead)
    
    # Move model to GPU
    if config.gpu >= 0:
        torch.cuda.set_device(config.gpu)
        model.cuda()

    # Create optimizer
    optim = torch.optim.SGD(
        model.parameters(), config.lr, momentum=config.momentum, weight_decay=config.wd)
    scheduler = ExponentialLR(optim, config.gamma)

    # Set up plotting
    best_score = float('inf')

    # Main training loop
    for epoch in range(config.epochs):

        tqdm.write('\n\n Beginning epoch {} of {}'.format(epoch, config.epochs))
        
        # Decay learning rate
        scheduler.step()

        # Train the model for one epoch
        train_loss, train_rms = run_epoch(
            epoch, train_loader, model, config, tensorboard, optim)

        tqdm.write('train epoch {}: train loss: {:.6f}\ttrain RMS: {:.2f}\tLR: {:.3f}'.format(epoch, train_loss, train_rms, optim.param_groups[0]['lr']))

        # Evaluate model on the validation set
        with torch.no_grad():
            val_loss, val_rms = run_epoch(
                epoch, val_loader, model, config, tensorboard)

        tqdm.write('val epoch {}: val loss: {:.6f}\tval RMS: {:.2f}'.format(epoch, val_loss, val_rms))

        # Record metrics and plot
        if tensorboard is not None:
            
            tensorboard.add_scalar('train/loss', train_loss, epoch)
            tensorboard.add_scalar('train/rms_error', train_rms, epoch)
            tensorboard.add_scalar('train/optimizer/lr', optim.param_groups[0]['lr'], epoch)

            tensorboard.add_scalar('val/loss', val_loss, epoch)
            tensorboard.add_scalar('val/rms_error', val_rms, epoch)
            
            # Save metrics
            aopwc.write_csv(os.path.join(logdir, 'metrics.csv'), epoch=epoch,
                train_loss=train_loss, train_rms=train_rms, val_loss=val_loss,
                val_rms=val_rms)
            
            # Save checkpoints
            best_score = min(val_rms, best_score)
            aopwc.save_checkpoint(os.path.join(logdir, 'latest.pth'),
                epoch, model, optim, best_score)
            if best_score == val_rms:
                aopwc.save_checkpoint(os.path.join(logdir, 'best.pth'),
                    epoch, model, optim, best_score)

    print('\n=== Training complete! ===')
    if logdir is not None:
        aopwc.save_checkpoint(os.path.join(logdir, 'final.pth'),
            epoch, model, best_score=best_score)
        
if __name__ == '__main__':
    main()
