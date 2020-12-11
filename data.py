import torch
import numpy as np

from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split


class DatasetGZSL(Dataset):
    """
    Class for the loading of ZSL Datasets.
    """

    def __init__(self, dataset='AWA2', generalized=False, device='cpu', purpose='test'):
        data = torch.load('datasets/' + dataset + '/data.pt')
        info = torch.load('datasets/' + dataset + '/info.pt')
        idx  = torch.load('datasets/' + dataset + '/splits.pt')

        # Load data
        self.features = MinMaxScaler().fit_transform(data['features'])
        self.features = torch.from_numpy(self.features).float().to(device)
        self.labels   = data['labels'].long().to(device)

        # Load info
        self.classes    = [names[0].replace('+', ' ') for names in info['classes']]
        self.attributes = info['attributes'].float().to(device)

        self.cnn_dim = self.features.shape[1]
        self.att_dim = self.attributes.shape[1]

        # Load splits
        self.train_idx       = idx['train'].astype(int)
        self.trainval_idx    = idx['trainval'].astype(int)
        self.val_idx         = idx['val'].astype(int)
        self.test_seen_idx   = idx['test_seen'].astype(int)
        self.test_unseen_idx = idx['test_unseen'].astype(int)

        # Unofficial splits for the purpose of validation
        if generalized == True and purpose == 'validate':
            if 'train_80' not in idx or 'train_20' not in idx:
                idx['train_80'], idx['train_20'] = train_test_split(self.train_idx, train_size=0.8)
                torch.save(idx, 'datasets/' + dataset + '/splits.pt')
            
            self.trainval_idx  = idx['train_80']
            self.test_seen_idx = idx['train_20']
            self.test_unseen_idx = self.val_idx
        
        if generalized == False and purpose == 'validate':
            self.trainval_idx  = idx['train']
            self.test_seen_idx = None
            self.test_unseen_idx = self.val_idx

        if generalized == False and purpose == 'test':
            self.trainval_idx  = np.concatenate([self.train_idx, self.val_idx])
            self.test_seen_idx = None

        self.seen_classes   = np.unique(self.labels[self.trainval_idx].to('cpu'))
        self.unseen_classes = np.unique(self.labels[self.test_unseen_idx].to('cpu'))
    
    def transfer_features(self, num_features):
        for c in self.unseen_classes:
            idx = [i for (i, v) in enumerate(self.test_unseen_idx) if self.labels[v] == c]
            idx = np.random.choice(idx, num_features)
            
            self.trainval_idx    = np.insert(self.trainval_idx, -1, self.test_unseen_idx[idx])
            self.test_unseen_idx = np.delete(self.test_unseen_idx, idx)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return [self.features[idx, :], self.attributes[self.labels[idx], :], self.labels[idx]]


class EmbeddingsDataset(Dataset):
    def __init__(self, dataset, model, num_seen, num_unseen):
        cuda = model.device.startswith('cuda')

        # Get seen embeddings from cnn features
        features, _, labels = dataset[dataset.trainval_idx]
        if cuda:
            labels = labels.to('cpu')

        z_cnns = []
        l_cnns = []
        
        for c in dataset.seen_classes:
            idx = np.where(labels == c)[0]
            idx = np.random.choice(idx, num_seen, replace=True)

            x_cnn = features[idx]
            l_cnn = labels[idx]

            z_cnn, _, _ = model.cnn_encoder(x_cnn)
            
            z_cnns.append(z_cnn)
            l_cnns.append(l_cnn)
        
        z_cnns = torch.cat(z_cnns)
        l_cnns = torch.cat(l_cnns)

        # Get unseen embeddings from attributes
        _, attributes, labels = dataset[dataset.test_unseen_idx]
        if cuda:
            labels = labels.to('cpu')

        z_atts = []
        l_atts = []

        for c in dataset.unseen_classes:
            idx = np.where(labels == c)[0]
            idx = np.random.choice(idx, num_unseen, replace=True)

            x_att = attributes[idx]
            l_att = labels[idx]

            mu  = model.mu(x_att)
            std = torch.exp(0.5 * model.logvar(x_att))
            eps = torch.randn_like(std)

            z_att = mu + eps*std
            
            z_atts.append(z_att)
            l_atts.append(l_att)
        
        z_atts = torch.cat(z_atts)
        l_atts = torch.cat(l_atts)

        self.embeddings = torch.cat([z_cnns, z_atts]).to(model.device)
        self.labels     = torch.cat([l_cnns, l_atts]).to(model.device)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if type(idx) is np.ndarray:
            idx = idx.tolist()

        return [self.embeddings[idx, :], self.labels[idx]]
