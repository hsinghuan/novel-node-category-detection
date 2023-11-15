import torch.nn as nn

class Model(nn.Module):
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier

    def reset_parameters(self):
        self.encoder.reset_parameters()
        self.classifier.reset_parameters()

    def forward(self, x, edge_index):
        return self.classifier(self.encoder(x, edge_index))

    def get_encoder_classifier(self):
        return self.encoder, self.classifier