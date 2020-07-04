import torch
from torch import nn, optim

from models import CSVaDE
from data import DatasetGZSL
from train import *

from utils import *
from argparse import ArgumentParser


def main(opt):
    if opt.name.endswith('.json'):
        # Run from config file
        configs = create_configs(opt.name)

        for (i, config) in enumerate(configs, start=1):
            # Options
            model_name = get_model_name(config)
            dataset = DatasetGZSL(config['general']['dataset'], opt.device, purpose='validate')
            tensorboard_dir = 'tensorboards/experiments/' + opt.name.split('/')[-1].split('.')[0]

            if config['general']['num_shots'] > 0:
                dataset.transfer_features(config['general']['num_shots'])

            # Init model
            model = CSVaDE(model_name,
                           cnn_dim=dataset.cnn_dim,
                           att_dim=dataset.att_dim,
                           num_classes=len(dataset.classes),
                           device=opt.device,
                           load_pretrained=opt.load_pretrained,
                           reset_classifier=opt.reset_classifier,
                           **config['architecture'])

            print('{:2d}) {}'.format(i, model_name))

            # Train embeddings
            for param in model.classifier.parameters():
                param.requires_grad = False

            if 'optimizer' not in config['embeddings']:
                optimizer = optim.Adam(model.parameters(), lr=0.0001)
            else:
                name = config['embeddings']['optimizer']['name']
                settings = config['embeddings']['optimizer']['settings']
                optimizer = str_to_class(name)(model.parameters(), **settings)
                del config['embeddings']['optimizer']

            criterion = {
                'function': nn.L1Loss(reduction='sum'),

                'coefficients': {
                    'beta':  {'start': 1,  'stop': 10, 'value': 1},
                }
            }

            hist = train_embeddings(model, dataset, optimizer, criterion, **config['embeddings'],
                                    verbose=False, tensorboard_dir=tensorboard_dir)
            print('\tLoss = {:.2f}'.format(hist[-1]))

            # Train classifier
            for param in model.parameters():
                param.requires_grad = not param.requires_grad

            optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=[0.5, 0.999])

            acc = train_classifier(model, dataset, optimizer, nn.NLLLoss(), **config['classifier'],
                                   verbose=False, tensorboard_dir=tensorboard_dir)

            print('\tH-acc = {:.2f}'.format(acc))

            # Train SVM
            acc = train_svm(model, dataset, verbose=False)
            print('\tH-acc = {:.2f}\n'.format(acc))
    else:
        # Run with terminal arguments
        dataset = DatasetGZSL(opt.dataset, opt.device)

        if opt.num_shots > 0:
            dataset.transfer_features(opt.num_shots)

        model = CSVaDE(opt.name,
                       cnn_dim=dataset.cnn_dim,
                       att_dim=dataset.att_dim,
                       num_classes=len(dataset.classes),
                       device=opt.device,
                       load_pretrained=opt.load_pretrained,
                       reset_classifier=opt.reset_classifier)

        # Train embeddings
        for param in model.classifier.parameters():
            param.requires_grad = False

        optimizer = optim.Adam(model.parameters(), lr=0.0001)

        criterion = {
            'function': nn.L1Loss(reduction='sum'),

            'coefficients': {
                'beta':  {'start': 1,  'stop': 10, 'value': 1},
            }
        }

        train_embeddings(model, dataset, optimizer, criterion)

        # Train classifier
        if opt.classifier == 'softmax':
            for param in model.parameters():
                param.requires_grad = not param.requires_grad

            optimizer = optim.Adam(model.parameters(), lr=0.0001, betas=[0.5, 0.999])

            train_classifier(model, dataset, optimizer, nn.NLLLoss(), top_k_acc=opt.top_k_acc)
        else:
            train_svm(model, dataset, top_k_acc=opt.top_k_acc)


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains the CSVaDE model.')

    parser.add_argument('name')

    parser.add_argument('--dataset', '-d', default='AWA2')
    parser.add_argument('--num-shots', '-n', type=int, default=0)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--load-pretrained', action='store_true')
    parser.add_argument('--reset-classifier', action='store_true')

    parser.add_argument('--classifier', '-c', default='svm')
    parser.add_argument('--top-k-acc', '-k', type=int, default=1)

    opt = parser.parse_args()

    main(opt)
