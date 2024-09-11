import torch

from src.models.custom_unet import get_model_custom_unet
from src.models.deeplabv3_resnet50 import get_model_deeplabv3_resnet50
from src.models.fcn_resnet50 import get_model_fcn_resnet50


def get_model(model_name):
    if model_name == 'deeplabv3_resnet50':
        return get_model_deeplabv3_resnet50()
    elif model_name == 'fcn_resnet50':
        return get_model_fcn_resnet50()
    elif model_name == 'custom_unet':
        return get_model_custom_unet(3, 1)
    else:
        raise ValueError(f'Unknown model: {model_name}')

def load_model_state(model_name, path):
    model = get_model(model_name)
    model.load_state_dict(torch.load(path))
    return model