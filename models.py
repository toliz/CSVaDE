import os
import torch
import numpy as np

from torch import nn
from utils import init_weights


class Encoder(nn.Module):
    def __init__(self, layer_sizes, device='cpu'):
        super(Encoder, self).__init__()

        layers = []
        for (in_features, out_features) in zip(layer_sizes[:-2], layer_sizes[1:-1]):
            layers.append( nn.Linear(in_features, out_features) )
            layers.append( nn.ReLU() )

        self.encoder = nn.Sequential(*layers)
        self.mu      = nn.Linear(layer_sizes[-2], layer_sizes[-1])
        self.logvar  = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        self.to(device)
        self.apply(init_weights)

    def forward(self, x):
        # Encode
        mu     = self.mu(self.encoder(x))
        logvar = self.logvar(self.encoder(x))

        # Reparametrize
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)

        z = mu + eps*std

        return z, mu, logvar


class Decoder(nn.Module):
    def __init__(self, layer_sizes, device='cpu'):
        super(Decoder, self).__init__()

        layers = []
        for (in_features, out_features) in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append( nn.Linear(in_features, out_features) )
            layers.append( nn.ReLU() )

        self.decoder = nn.Sequential(*layers[:-1]) # We don't need the last ReLU

        self.to(device)
        self.apply(init_weights)

    def forward(self, z):
        return self.decoder(z)


class Classifier(nn.Module):
    def __init__(self, in_features, out_features, device='cpu'):
        super(Classifier, self).__init__()

        self.linear = nn.Linear(in_features, out_features)

        self.to(device)
        self.apply(init_weights)

    def forward(self, x):
        return nn.functional.log_softmax(self.linear(x), dim=1)


class CSVaDE(nn.Module):
    def __init__(self, name, cnn_dim, att_dim, embeddings_dim, num_classes,
                 cnn_hidden_layers=[1000], att_hidden_layers=[1000],
                 device='cpu', load_pretrained=True, reset_classifier=False):
        super(CSVaDE, self).__init__()

        self.name   = name
        self.device = device

        self.mu     = torch.nn.Parameter(torch.zeros(num_classes, embeddings_dim, device=device))
        self.logvar = torch.nn.Parameter(torch.zeros(num_classes, embeddings_dim, device=device))
        
        self.cnn_encoder = Encoder([cnn_dim,        *cnn_hidden_layers, embeddings_dim], device)
        self.cnn_decoder = Decoder([embeddings_dim, *cnn_hidden_layers, cnn_dim       ], device)
        self.att_encoder = Encoder([att_dim,        *att_hidden_layers, embeddings_dim], device)
        self.att_decoder = Decoder([embeddings_dim, *att_hidden_layers, att_dim       ], device)
        
        self.classifier  = Classifier(embeddings_dim, num_classes, device)

        self.embeddings_history =  np.array([])
        self.classifier_history = [np.array([]) for _ in range(4)] # history for classifier loss and top-1 accuracies

        if load_pretrained:
            self.load_pretrained('saved/' + name + '.pt', reset_classifier)

    def forward(self, x):
        _, z, _ = self.cnn_encoder(x)   # Sample without reparametrization
        
        return self.classifier(z)
    
    def load_pretrained(self, path, reset_classifier=False):
        if not os.path.exists(path):
            print('Path \'{}\' for pretrained self doesn\'t exist!'.format(path))
            return
        
        print('Loading pretrained self from: {}\n'.format(path))
        
        checkpoint = torch.load(path, map_location=self.device)
        
        self.load_state_dict(checkpoint['state_dict'])

        self.embeddings_history = checkpoint['embeddings_history']
        self.classifier_history = checkpoint['classifier_history']

        if reset_classifier:
            self.classifier.apply(init_weights)
            self.classifier_history = [np.array([]) for _ in range(4)]
