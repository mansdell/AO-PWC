"""
Trains a Predictive Wavefront Model from scratch
"""

import os
import csv
import math
import time
from argparse import ArgumentParser
from progressbar import ProgressBar

# If we're running on a remote server with out graphics, use the Agg backend
if 'DISPLAY' not in os.environ:
    import matplotlib
    matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader

import aopwc

def run_epoch(dataloader, model, config, optimizer=None):
    
    # Create progress bar
    progress = ProgressBar()

    # Initialize metrics
    total_loss = 0
    total_sqr_error = 0

    # Set model to training or validation mode
    model.train() if optimizer is not None else model.eval()

    # Iterate over examples in the dataset
    for wavefront in progress(dataloader):

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

    # Print epoch summary
    print(' - Loss: {:.2e}\n - RMS error: {:.2f}nm\n'.format(
        avg_loss, rms_error))

    return avg_loss, rms_error


def plot(metrics, plot_name, logdir):
    fig = plt.figure(plot_name)
    fig.clear()

    train_metrics, val_metrics = zip(*metrics)
    plt.plot(train_metrics, label='train')
    plt.plot(val_metrics, label='val')

    plt.xlabel('epochs')
    plt.ylabel(plot_name)
    plt.title(plot_name)
    plt.legend()

    if logdir is not None:
        plt.savefig(os.path.join(logdir, '{}.pdf'.format(plot_name)))
   

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
    
    return parser.parse_args()


def main():

    # Parse command line arguments
    config = parse_args()

    # Create directory for the experiment and save the config
    logdir = create_experiment(config)

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
    if config.arch == 'ConvLSTM':
        model = aopwc.ConvLSTM(config.hidden)
    
    # Move model to GPU
    if config.gpu >= 0:
        torch.cuda.set_device(config.gpu)
        model.cuda()

    # Create optimizer
    optim = torch.optim.SGD(
        model.parameters(), config.lr, momentum=0.9, weight_decay=1e-4)

    # Set up plotting
    plt.ion()
    loss_metrics = list()
    rms_metrics = list()
    best_score = float('inf')

    # Main training loop
    for epoch in range(config.epochs):
        print('\n=== Starting epoch {} of {} ===\n'.format(
            epoch+1, config.epochs))

        # Train the model for one epoch
        print('Training')
        train_loss, train_rms = run_epoch(train_loader, model, config, optim)

        # Evaluate model on the validation set
        print('Validating')
        with torch.no_grad():
            val_loss, val_rms = run_epoch(val_loader, model, config)

        # Record metrics and plot
        loss_metrics.append((train_loss, val_loss))
        rms_metrics.append((train_rms, val_rms))

        if logdir is not None:

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
        
        # Plot and save figures
        plot(loss_metrics, 'loss', logdir)
        plot(rms_metrics, 'rms_error', logdir)
        plt.draw()
        plt.pause(0.01)


    print('\n=== Training complete! ===')
    aopwc.save_checkpoint(os.path.join(logdir, 'final.pth'),
        epoch, model, best_score=best_score)
        
if __name__ == '__main__':
    main()