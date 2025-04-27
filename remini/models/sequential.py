class Sequential:
    def __init__(self, *layers):
        self.layers = layers

    def forward(self, x):
        for layer in self.layers:
            x = layer.forward(x)
            x = x.leaky_relu(negative_slope=0.01)
        return x

    def parameters(self):
        params = []
        for layer in self.layers:
            params.append(layer.weights)
            params.append(layer.bias)
        return params
