import curaitor.resnet_model
import torch


def initialize_model(path_to_checkpoint):
    model = curaitor.resnet_model.ResNet()
    state = torch.load(path_to_checkpoint)
    model.load_state_dict(state)
    model.freeze()
    return model
