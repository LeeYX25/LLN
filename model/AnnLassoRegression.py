import torch
import torch.nn as nn
import numpy as np
import utils
from torch.nn.functional import linear, normalize
from .ModelArchitecture import ModelArchitecture

class AnnLassoRegression(ModelArchitecture):
    def __init__(self, penalty=0, p2=20, lambda_qut=None, device=None):
        super().__init__(penalty = penalty, model_type = 0, p2=p2, lambda_qut=lambda_qut, device=device)

    def forward(self, X):
        layer1_output = self.act_fun(self.layer1(X))
        w2_weights_normalized = normalize(self.layer2.weight, p=2, dim=1)
        logits = linear(layer1_output, w2_weights_normalized, self.layer2.bias)
        return logits.squeeze()
