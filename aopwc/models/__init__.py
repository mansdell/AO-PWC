from .convrnn import ConvLSTM
from .prednet import PredNet
from .tip_tilt import TipTiltNetwork

def make_model(config):
    """
    Creates a model from a config object

    See the command line arguments in `scripts/train.py` for details of the 
    attributes in `config`.
    """
    if config.arch == 'ConvLSTM':
        model = ConvLSTM(config.hidden)
    
    return model