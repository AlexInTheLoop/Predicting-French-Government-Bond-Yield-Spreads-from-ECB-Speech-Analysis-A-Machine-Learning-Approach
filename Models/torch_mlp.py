import torch.nn as nn
from Models.torch_base import TorchBaseClassifier

class MLPNet(nn.Module):
    def __init__(self, input_dim, hidden_dims=(512, 128), dropout=0.2, num_classes=3):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

class MLPClassifier(TorchBaseClassifier):
    def __init__(self, input_dim, num_classes=3, hidden_dims=(512, 128), dropout=0.2,
                 lr=1e-3, weight_decay=0.0, batch_size=64, epochs=20,
                 device=None, early_stopping_patience=5, verbose=True):
        model = MLPNet(input_dim, hidden_dims=hidden_dims, dropout=dropout, num_classes=num_classes)
        super().__init__(model, name="MLP", lr=lr, weight_decay=weight_decay, batch_size=batch_size,
                         epochs=epochs, device=device, early_stopping_patience=early_stopping_patience,
                         verbose=verbose)