import torch
import torch.nn as nn
import torch.nn.functional as F

from .convrnn import ConvLSTMCell

class PredNet(nn.Module):
    
    def __init__(self, R_channels, A_channels, steps_ahead=1):
        
        assert len(R_channels) == len(A_channels)
        self.R_channels = R_channels + [0]
        self.A_channels = A_channels
        self.num_layers = len(self.A_channels)

        self.conv_R = nn.ModuleList()
        self.conv_A_hat = nn.ModuleList()
        self.conv_A = nn.ModuleList()
        for l in range(self.num_layers):

            # Recurrent representation layer R_l
            in_chls = 2 * self.A_channels[l] + self.R_channels[l+1]
            self.conv_R.append(ConvLSTMCell(in_chls, self.R_channels[l]))

            # Target layer A_l
            self.conv_A.append(nn.Conv2d(
                self.A_channels[l], self.A_channels[l], kernel_size=3,
                padding=1))
            
            # Prediction layer A_hat_l
            if l < self.num_layers - 1:
                self.conv_A_hat.append(nn.Conv2d(
                    self.R_channels[l], self.A_channels[l+1], kernel_size=3, 
                    stride=2, padding=1))
            

    def forward(self, sequence):
        
        # Initialize the representation and error vectors at time t=0
        batch_size, _, time_steps, height, width = sequence.size()
        rep, error, cell = list(), list(), list()
        for l in range(self.num_layers):

            rep.append(sequence.new_zeros(
                batch_size, self.R_channels[l], height, width))
            error.append(sequence.new_zeros(
                batch_size, 2 * self.A_channels[l], time_steps, height, width))
            cell.append(None)
            height, width = int(height / 2), int(width / 2)

        predictions = torch.zeros_like(sequence)

        # Iterate over frames in the sequence
        for t, frame in enumerate(torch.unbind(sequence, dim=2)):
            
            for l in range(self.num_layers-1, -1, -1):
                conv_R = self.conv_R[l]
                if l == self.num_layers - 1: 
                    rep[l], cell[l] = conv_R(error[l][:, :, t], rep[l], cell[l])
                else:
                    previous = F.upsample(rep[l+1], size=error[l].shape[-2:])
                    inpt = torch.cat((error[l][:, : ,t], previous), dim=-1)  
                    rep[l], cell[l] = conv_R(inpt, rep[l], cell[l])
            
            A = frame
            for l in range(self.num_layers):
                
                # Make prediction
                A_hat = F.relu(self.conv_A_hat(rep[l]))
                if l == 0:
                    predictions[:, :, t] = A_hat
                
                # Compute error
                error[l][:, :, t] = torch.cat(
                    [F.relu(A_hat - A), F.relu(A - A_hat)], 1)

                # Update A for the next layer
                if l < self.num_layers - 1:
                    A = F.relu(self.conv_A[l](error[l]))
        

        return predictions, error

            

            

                

