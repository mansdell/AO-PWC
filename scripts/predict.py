import os
import math
import time
import numpy as np
from numpy import ma
from argparse import ArgumentParser
import matplotlib.pyplot as plt
import torch
import aopwc

COLORMAP = 'inferno'
VMIN = aopwc.WAVEFRONT_MEAN - 2 * aopwc.WAVEFRONT_STD
VMAX = aopwc.WAVEFRONT_MEAN + 2 * aopwc.WAVEFRONT_STD
    

def main():
    parser = ArgumentParser()
    parser.add_argument('name', type=str, 
                        help='name of experiment to load')
    parser.add_argument('--logdir', '-d', type=str, default='experiments',
                        help='location where experiments are stored')
    parser.add_argument('--gpu', '-g', type=int, default=0,
                        help='index of current gpu (use -1 for cpu inference)')
    args = parser.parse_args()

    # Load config file
    logdir = os.path.join(args.logdir, args.name)
    config = aopwc.load_config(logdir)

    # Build model
    model = aopwc.make_model(config)

    # Load model checkpoint
    aopwc.load_checkpoint(os.path.join(logdir, 'latest.pth'), model)
    
    # Move model to GPU
    if args.gpu >= 0:
        torch.cuda.set_device(args.gpu)
        model.cuda()
    model.eval()
    
    # Load dataset
    # TODO should really only run inference over the validation set
    dataset = aopwc.WavefrontDataset('data/phase_screens_part1')

    # Setup matplotlib
    fig, (gt_ax, pred_ax) = plt.subplots(ncols=2, figsize=(12, 4))
    cbar_ax = None
    plt.ion()

    # Iterate over examples in the dataset
    for i, wavefront in enumerate(dataset):

        # Add a batch dimension
        wavefront = wavefront.unsqueeze(0)
        if args.gpu >= 0:
            wavefront = wavefront.cuda()
        
        # Run inference on model
        with torch.no_grad():
            prediction = model(aopwc.remove_nans(wavefront))
        
        # Un-normalize
        wavefront = wavefront * aopwc.WAVEFRONT_STD + aopwc.WAVEFRONT_MEAN
        prediction = prediction * aopwc.WAVEFRONT_STD + aopwc.WAVEFRONT_MEAN 
        
        # Compute RMS error over the current sequence
        sqr_error = aopwc.masked_l2_loss(
            prediction, wavefront, config.steps_ahead)
        rms_error = math.sqrt(float(sqr_error))
        
        # Plot predictions for each frame
        for t in range(0, wavefront.size(1), 10):
            
            # Convert torch tensors to numpy arrays
            pred_frame = prediction[0, t].cpu().numpy()
            gt_frame = wavefront[0, t].cpu().numpy()
            mask = np.isnan(gt_frame)

            # Set figure title
            fig.suptitle('Cube #{}  $E_{{RMS}}$ = {:.1f}nm  ' \
                         'Elapsed time = {:2d}ms'.format(i, rms_error, t))

            # Visualize original frame
            gt_ax.clear()
            gt_ax.set_title('Original')
            gt_ax.imshow(ma.masked_array(gt_frame, mask=mask),
                         cmap=COLORMAP, vmin=VMIN, vmax=VMAX)
            gt_ax.axis('off')
            
            # Visualize predicted frame
            pred_ax.clear()
            pred_ax.set_title('Predicted')
            p = pred_ax.imshow(ma.masked_array(pred_frame, mask=mask),
                           cmap=COLORMAP, vmin=VMIN, vmax=VMAX)
            pred_ax.axis('off')
        
            # Add colorbar
            cbar = plt.colorbar(p, ax=(gt_ax, pred_ax), cax=cbar_ax)
            cbar.set_label('Phase offset (nm)')
            cbar_ax = cbar.ax
            
            # Draw figures
            plt.draw()
            plt.pause(0.01)
            time.sleep(0.1)
            

if __name__ == '__main__':
    main()