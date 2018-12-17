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

# If we're running on a remote server with out graphics, use the Agg backend

import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from tensorboardX import SummaryWriter

import aopwc

def run_epoch(dataloader, model, config, optimizer=None):
    # Initialize metrics
    total_loss = 0
    total_sqr_error = 0

    # Set model to training or validation mode
    model.train() if optimizer is not None else model.eval()

    # Iterate over examples in the dataset
    for wavefront in tqdm(dataloader):

        # Move inputs to GPU
        if config.gpu >= 0:
            wavefront = wavefront.cuda()

        # Run model to generate predicted wavefronts
        predictions = model(wavefront)

        # Compute loss
        loss = aopwc.masked_l1_loss(predictions, wavefront, config.steps_ahead)
        sqr_error = aopwc.masked_l2_loss(predictions, wavefront, 
                                         config.steps_ahead)
        total_loss += float(loss)
        total_sqr_error += float(sqr_error)

        # Update parameters if we are in training mode
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Compute average loss
    count = len(dataloader)
    avg_loss = total_loss / count
    rms_error = math.sqrt(total_sqr_error / count) * aopwc.WAVEFRONT_STD

    return avg_loss, rms_error

def create_experiment(config):
    # Print config information
    print('-' * 31 + '\n--- Starting new experiment ---\n' + '-' * 31)
    for key, value in config.__dict__.items():
        print('{}: {}'.format(key, value))
    
    if config.name is None:
        print('\nWARNING: no name specified for experiments: ' \
              'results will not be saved')
        return None
    
    # Check that there isn't already an experiment with the given name
    logdir = os.path.join(config.logdir, config.name)
    assert not os.path.exists(logdir), 'experiment \'{}\' already exists' \
        ' at location {}'.format(config.name, config.logdir)
    
    # Create the experiment
    print('\nExperiment directory:\n  {}'.format(logdir))
    os.makedirs(logdir)

    # Save the config data to a YAML file
    aopwc.save_config(logdir, config)
    
    return logdir


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--name', '-n', type=str, default=None,
                        help='name of the experiment. If left blank,'
                             + ' no logging information will be saved to disk')
    parser.add_argument('--logdir', '-d', type=str, default='experiments',
                        help='location to store experiment log files')
    parser.add_argument('--batch-size', '-b', type=int, default=24,
                        help='number of examples per mini-batch')
    parser.add_argument('--lr', '-l', type=float, default=1e-2,
                        help='learning rate')
    parser.add_argument('--epochs', '-e', type=int, default=100,
                        help='number of training epochs')
    parser.add_argument('--steps-ahead', '-s', type=int, default=2,
                        help='number of timesteps into the future to predict')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='index of current gpu (use -1 for cpu training)')
    parser.add_argument('--arch', '-a', type=str, default='ConvLSTM',
                        choices=['ConvLSTM'],
                        help='name of model architecture')
    parser.add_argument('--hidden', type=int, nargs='+',
                        default=[16, 32, 64, 64],
                        help='number of feature channels in hidden layers')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='number of worker threads for data loading'
                             + ' (use 0 for single-threaded mode)')
    parser.add_argument('--train-split', type=float, default=0.8,
                        help='fraction of data used for training')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='fraction of data used for validation')
    parser.add_argument('--schedule', type=int, nargs='*', default=[50, 150],
                        help='decrease lr by a factor of 10 after this many' \
                             ' epochs')
    return parser.parse_args()


def main():

    # Parse command line arguments
    config = parse_args()

    # Create directory for the experiment and save the config
    logdir = create_experiment(config)

    # Create tensorboard summary writer
    dir_suffix = time.strftime("%Y-%m-%d-%H-%M-%S") + '-' + platform.node()
    print('Creating tensorboard summary...')
    log_dir = os.path.join(logdir, dir_suffix)
    tensorboard = SummaryWriter(log_dir=logdir, comment='')

    # Get current GIT SHA tag
    repo = git.Repo(search_parent_directories=True)
    git_sha = repo.head.object.hexsha

    # Output debug stuff to tensorboard strings
    tensorboard.add_text('command_line', ' '.join(sys.argv), 0)
    tensorboard.add_text('git_sha', str(git_sha), 0)
    tensorboard.add_text('config', str(config.__dict__), 0)

    # Create dataset
    dataset = aopwc.WavefrontDataset('./data/phase_screens_part1')

    # Split dataset into train and validation sets
    train_data, val_data = aopwc.split_dataset(
        dataset, (config.train_split, config.val_split), shuffle=False)

    # Create dataloaders
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, 
                              num_workers=config.workers)
    val_loader = DataLoader(val_data, config.batch_size, shuffle=True, 
                            num_workers=config.workers)

    # Build model
    model = aopwc.make_model(config)
    
    # Move model to GPU
    if config.gpu >= 0:
        torch.cuda.set_device(config.gpu)
        model.cuda()

    # Create optimizer
    optim = torch.optim.SGD(
        model.parameters(), config.lr, momentum=0.9, weight_decay=1e-4)
    scheduler = MultiStepLR(optim, config.schedule)

    # Set up plotting
    best_score = float('inf')

    # Main training loop
    for epoch in range(config.epochs):
        print('\n=== Starting epoch {} of {} ===\n'.format(
            epoch+1, config.epochs))

        # Decay learning rate
        scheduler.step()

        # Train the model for one epoch
        print('Training')
        train_loss, train_rms = run_epoch(train_loader, model, config, optim)

        # Evaluate model on the validation set
        print('Validating')
        with torch.no_grad():
            val_loss, val_rms = run_epoch(val_loader, model, config)

        # Record metrics and plot
        if logdir is not None:
            tensorboard.add_scalar('train/loss', train_loss, epoch)
            tensorboard.add_scalar('train/rms_error', train_rms, epoch)
            tensorboard.add_scalar('train/optimizer/lr', optim.param_groups[0]['lr'], epoch)

            tensorboard.add_scalar('val/loss', val_loss, epoch)
            tensorboard.add_scalar('val/rms_error', val_rms, epoch)
            
            # if self.verbose_tensorboard:
            #     self.tensorboard.add_histogram('train/data', data[:], n_iter, bins='auto')
            #     self.tensorboard.add_histogram('train/label', target, n_iter, bins='auto')
            #     data_video = torch.unsqueeze(data, dim=1)
            #     self.tensorboard.add_video('train/tpf_video', vid_tensor=data_video)

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
        
        tqdm.write('train epoch {}: train loss: {:.6f}\tval loss: {:.6f}\ttrain RMS: {:.2f}\tval RMS: {:.2f}\tLR: {:.3f}'.format(epoch, train_loss, val_loss, train_rms, val_rms, optim.param_groups[0]['lr']))


    print('\n=== Training complete! ===')
    aopwc.save_checkpoint(os.path.join(logdir, 'final.pth'),
        epoch, model, best_score=best_score)
        
if __name__ == '__main__':
    main()