import torch
import torch.nn as nn
import torch.nn.functional as F
from ..utils import remove_nans

class ConvLSTMCell(nn.Module):
    """
    Implementation of a single Long Short-Term Memory unit.

    Args:
        input_channels (int): number of input features
        hidden_channels (int): number of output features
        kernel_size (int or tuple): height and width of 2D convolutional kernel
        padding (int or tuple): amount of padding to add to the top/bottom and
            left/right of the image. Use `(kernel_size-1)/2` to make the output
            size the same as the input
        bias (bool): if true, adds a learnable bias to the output

    See the pytorch docs on convolution and LSTMs for more details:
        https://pytorch.org/docs/stable/nn.html#conv2d
        https://pytorch.org/docs/stable/nn.html#lstmcell
    """

    def __init__(self, input_channels, hidden_channels, kernel_size=3, 
                 padding=1, bias=True):
        super(ConvLSTMCell, self).__init__()

        self.hidden_chls = hidden_channels 
        self.conv = nn.Conv2d(input_channels + hidden_channels, 
                              hidden_channels * 4,
                              kernel_size, padding, bias)
                            
    def forward(self, input, hidden=None, cell_state=None):

        # Initialize cell and hidden state on the first iteration
        if hidden is None or cell_state is None:
            batch, _, height, width = input.size()
            hidden = input.new_zeros(batch, self.hidden_chls, height, width)
            cell_state = input.new_zeros(batch, self.hidden_chls, height, width)
        
        # Compute all gates in parallel
        input_gate, forget_gate, output_gate, cell_update = \
            torch.chunk(self.conv(torch.cat([input, hidden], dim=1)), 4, dim=1)

        # Apply non-linearities
        input_gate = F.sigmoid(input_gate)
        forget_gate = F.sigmoid(forget_gate)
        output_gate = F.sigmoid(output_gate)
        cell_update = F.tanh(cell_update)

        # Update cell and hidden state
        cell_state = forget_gate * cell_state + input_gate * cell_update
        hidden = output_gate * F.tanh(cell_state)

        return hidden, cell_state


class ConvLSTM(nn.Module):
    """
    A simple stacked convolutional LSTM model.

    Args:
        hidden_channels (list): containing the output feature size for each of
            the LSTM layers 
        steps_ahead (int): number of steps into the future to predict
    """
    def __init__(self, hidden_channels, steps_ahead=1):
        super(ConvLSTM, self).__init__()
        self.steps_ahead = steps_ahead

        # Create LSTM layers
        in_channels = 1
        self.layers = nn.ModuleList()
        for out_channels in hidden_channels:
            self.layers.append(ConvLSTMCell(in_channels, out_channels))
            in_channels = out_channels

        # Final prediction layer
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=3, padding=1)
    

    def forward(self, sequence):
        
        # Initialize hidden state
        hidden = [None] * len(self.layers)
        cell = [None] * len(self.layers)

        # Initialize frames which have not yet been predicted
        predictions = [
            torch.zeros_like(sequence[:, 0]) for _ in range(self.steps_ahead)
        ]

        # Remove NaNs from input
        sequence = remove_nans(sequence[:, :-self.steps_ahead])

        # Iterate over frames 0 to T-n
        for frame in torch.unbind(sequence, dim=1):

            # Apply ConvLSTM layers
            x = frame.unsqueeze(1)
            for l, layer in enumerate(self.layers):
                hidden[l], cell[l] = layer(x, hidden[l], cell[l])
                x = hidden[l]
            
            # Generate final prediction
            output = self.conv(x).squeeze(1)
            predictions.append(output)

        # Combine predictions into a single tensor
        return torch.stack(predictions, dim=1) 
