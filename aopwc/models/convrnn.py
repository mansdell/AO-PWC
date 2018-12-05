import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvLSTMCell(nn.Module):

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
            batch, _, height, width, height = input.size()
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
