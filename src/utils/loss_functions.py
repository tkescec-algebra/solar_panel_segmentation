from torch import nn
import segmentation_models_pytorch as smp


def get_loss_function(fun_name):
    if fun_name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif fun_name == 'DiceLoss':
        return smp.losses.DiceLoss(mode='multiclass')
    elif fun_name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif fun_name == 'L1Loss':
        return nn.L1Loss()
    else:
        raise ValueError(f'Unknown loss function: {fun_name}')