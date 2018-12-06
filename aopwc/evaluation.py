import torch
import torch.nn.functional as F

def masked_l1_loss(prediction, target, steps_ahead=0, size_average=True):
    """
    Compute the mean absolute error (MAE) between two tensors

    Args:
        prediction (Tensor): predicted sequence
        target (Tensor): ground truth sequence
        steps_ahead (int): number of frames into the future to predict. The 
            first `steps_ahead' frames will be ignored in the loss calculation
        size_average (bool): returns the mean loss if True or the total loss
            if False
    
    Returns:
        loss (Tensor): a 0d scalar tensor containing the mean absolute error
    """

    # Ignore frames where there is no corresponding prediction
    prediction = prediction[:, steps_ahead:]
    target = target[:, steps_ahead:]

    # Weird Pytorch way of finding all the NaNs in the sequence
    mask = target != target

    # Compute pixelwise loss
    loss = F.l1_loss(prediction, target, reduce=False)

    # Take the mean over valid regions of the sequence
    if size_average:
        return loss[mask].mean()
    else:
        return loss[mask].sum()



def masked_l2_loss(prediction, target, steps_ahead=0, size_average=True):
    """
    Compute the mean squared error (MSE) between two tensors

    Args:
        prediction (Tensor): predicted sequence
        target (Tensor): ground truth sequence
        steps_ahead (int): number of frames into the future to predict. The 
            first `steps_ahead' frames will be ignored in the loss calculation
        size_average (bool): returns the mean loss if True or the total loss
            if False
    
    Returns:
        loss (Tensor): a 0d scalar tensor containing the mean squared error
    """

    # Ignore frames where there is no corresponding prediction
    prediction = prediction[:, steps_ahead:]
    target = target[:, steps_ahead:]

    # Ignore NaNs
    mask = target != target

    # Compute RMS
    sqr_error = (target - prediction) ** 2

    # Take mean or sum over valid values
    if size_average:
        return sqr_error[mask].mean()
    else:
        return sqr_error[mask].sum()