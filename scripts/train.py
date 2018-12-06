import os
import math
import time
from argparse import ArgumentParser
from progressbar import ProgressBar
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
    print('\nSummary:')
    print(' - Loss: {:.2e}\n - RMS error: {:.2}nm'.format(avg_loss, rms_error))

    return avg_loss, rms_error


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--batch-size', '-b', type=int, default=32,
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
                        options=['ConvLSTM'],
                        help='name of model architecture')
    parser.add_argument('--hidden', '-h', type=int, nargs='+',
                        default=[16, 32, 64, 64],
                        help='number of feature channels in hidden layers')
    parser.add_argument('--workers', '-w', type=int, default=4,
                        help='number of worker threads for data loading'
                             + ' (use 0 for single-threaded mode)')
    
    return parser.parse_args()


def main():
    
    # Parse command line arguments
    config = parse_args()

    # TODO Create directory for the experiment and save config
    # logdir = os.path.join('./experiments', ...)

    # Create dataset
    dataset = aopwc.WavefrontDataset('./data/phase_screens_part1')

    # Split dataset into train and validation sets
    train_data, val_data = aopwc.split_dataset(dataset, (0.8, 0.2), False)

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

    # Main training loop
    for epoch in range(config.max_epochs):
        print('== Starting epoch {} of {} =='.format(epoch, config.epochs))

        # Train the model for one epoch
        print('-- Training --')
        train_loss, train_rms = run_epoch(train_loader, model, config, optim)

        # Evaluate model on the validation set
        print('\n-- Validating --')
        val_loss, val_rms = run_epoch(val_loader, model, config)

        # TODO Save checkpoints
    
    print('\n=== Training complete! ===')
        
if __name__ == '__main__':
    main()