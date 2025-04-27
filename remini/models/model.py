# remini/models/model.py
from remini.layers.dense import Dense

class Model:
    def __init__(self):
        self.layers = [
            Dense(3, 2),  # Example Dense layer (3 inputs -> 2 outputs)
        ]

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params
