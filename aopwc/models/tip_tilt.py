import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalConvolution(nn.Conv1d):
    """
    1D conv which adds padding to ensure the output only depends on past inputs
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, bias=True):
        super(CausalConvolution, self).__init__(
            in_channels, out_channels, kernel_size)
        self.pad = kernel_size - 1
    
    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return super(CausalConvolution, self).forward(x)


class TipTiltNetwork(nn.Module):
    """
    1D convolutional network for predicting open-loop tip tilt positions
    """
    def __init__(self, hidden_channels, steps_ahead=1, mean=[-2.7362, 4.6941],
                 stdev=[2.4027, 1.5286]):
        
        super(TipTiltNetwork, self).__init__()
        self.steps_ahead = steps_ahead

        # Construct hidden layers
        in_channels = 2
        self.layers = nn.ModuleList()
        for out_channels in hidden_channels:
            self.layers.append(CausalConvolution(in_channels, out_channels))
            in_channels = out_channels
        
        # Add final output layer
        self.fc = nn.Conv1d(in_channels, 2, kernel_size=1)

        # Save mean and standard deviation
        self.register_buffer('mean', torch.tensor(mean).view(2, 1))
        self.register_buffer('stdev', torch.tensor(stdev).view(2, 1))
    

    def forward(self, input):

        # Normalize input
        x = (input - self.mean) / self.stdev

        # Apply hidden layers
        for layer in self.layers:
            x = F.relu(layer(x))

        # Predict x-y coordinates and unnormalize
        out = self.fc(x) * self.stdev + input

        # Align predictions with inputs
        return torch.cat(
            [input[:, :, :self.steps_ahead], out[:, :, :-self.steps_ahead]], 
            dim=-1)
