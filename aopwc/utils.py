import os
import csv
import torch
import yaml
from argparse import Namespace

def remove_nans(tensor, replace_with=0.):
    """
    Replace all NaNs in a tensor with the given value
    """
    mask = tensor == tensor
    output = torch.full_like(tensor, replace_with)
    output[mask] = tensor[mask]
    return output

def save_config(logdir, config):
    """
    Saves a config namespace object to a YAML file

    Args:
        logdir (str): location to save config
        config (Namespace): an argparse namespace containing the configuration
            for the current experiment
    """
    with open(os.path.join(logdir, 'config.yml'), 'w') as f:
        yaml.dump(config.__dict__, f)

def load_config(logdir):
    """
    Loads a config file as an argparse namespace
    """
    with open(os.path.join(logdir, 'config.yml'), 'r') as f:
        return Namespace(**yaml.load(f))


def write_csv(filename, **values):
    """
    Append a data point to a csv file
    """
    writeheader = not os.path.isfile(filename)
    fieldnames = sorted(values.keys())

    with open(filename, 'a') as f:
        writer = csv.DictWriter(f, fieldnames, dialect='excel-tab')
        if writeheader:
            writer.writeheader()
        writer.writerow(values)
    

def save_checkpoint(filename, epoch, model, optimizer=None, best_score=0):
    """
    Saves a training snapshot which can be used to resume training

    Args:
        filename (str): path to save checkpoint to
        epoch (int): the current training epoch
        model (nn.Module): a Pytorch model to save
        optimizer (nn.optim.Optimizer): the optimizer used for training
        best_score (float): the best score observed so far
    """
    print('Saving checkpoint \'{}\''.format(filename))
    torch.save({
        'model' : model.state_dict(),
        'optim' : optimizer.state_dict() if optimizer is not None else None,
        'epoch' : epoch,
        'best_score' : best_score
    }, filename)


def load_checkpoint(filename, model, optimizer=None):
    print('Loading checkpoint \'{}\''.format(filename))
    ckpt = torch.load(filename)

    # Load model weights
    model.load_state_dict(ckpt['model'])

    # Restore optimizer state
    if optimizer is not None:
        optimizer.load_state_dict(ckpt['optim'])
    
    return ckpt['epoch'], ckpt['best_score']
