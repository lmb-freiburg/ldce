from torch import nn

class DINOLinear(nn.Module):
    def __init__(self, dino, linear_classifier) -> None:
        super().__init__()
        self.dino = dino
        self.linear = linear_classifier
    
    def forward(self, x):
        x = self.dino(x)
        return self.linear(x)

class LinearClassifier(nn.Module):
    """Linear layer to train on top of frozen features"""
    def __init__(self, dim, num_labels=1000):
        super(LinearClassifier, self).__init__()
        self.num_labels = num_labels
        self.linear = nn.Linear(dim, num_labels)
        self.linear.weight.data.normal_(mean=0.0, std=0.01)
        self.linear.bias.data.zero_()

    def forward(self, x):
        # flatten
        x = x.view(x.size(0), -1)

        # linear layer
        return self.linear(x)