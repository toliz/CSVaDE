import os
import time
import torch
import shutil

from torch import nn, optim
from models import CSVaDE

from data import DatasetGZSL
from torch.utils.data import Subset, DataLoader

from utils import *
from torch.utils.tensorboard import SummaryWriter
from argparse import ArgumentParser


def batch_loss(model, cnn, att, labels, reconstruction_loss_function):
    z_cnn, mu_cnn, logvar_cnn = model.cnn_encoder(cnn)
    z_att, mu_att, logvar_att = model.att_encoder(att)
    
    cnn_recon = model.cnn_decoder(z_cnn)
    att_recon = model.att_decoder(z_att)

    mu     = model.mu[labels]
    logvar = model.logvar[labels]

    # VaDE Loss
    reconstruction_loss = reconstruction_loss_function(cnn_recon, cnn) + reconstruction_loss_function(att_recon, att)

    KLD = -0.5 * (torch.sum(1 + (logvar_cnn - logvar) - ((mu_cnn - mu).pow(2) + logvar_cnn.exp()) / logvar.exp()) + \
                  torch.sum(1 + (logvar_att - logvar) - ((mu_att - mu).pow(2) + logvar_att.exp()) / logvar.exp()))

    return [reconstruction_loss, KLD]


def train_embeddings(model,
                     dataset,
                     optimizer,
                     criterion,
                     num_epochs=100,
                     batch_size=50,
                     verbose=True,
                     tensorboard_dir='tensorboards/models/'):    
    # Sanity Check
    start_epoch = len(model.embeddings_history) + 1

    if start_epoch == num_epochs:
        return
    else:
        if verbose:
            print("\033[1mTraining VAE\033[0m\n")
        else:
            print('\tTraining VAE for model: ' + model.name)

    # Initialize writer for Tensorboard
    if tensorboard_dir != None:
        if tensorboard_dir[-1] != '/':
            tensorboard_dir += '/'
        
        if start_epoch == 1 and os.path.exists(tensorboard_dir + model.name):
            shutil.rmtree(tensorboard_dir + model.name)

        writer = SummaryWriter(tensorboard_dir + model.name, filename_suffix='.embeddings')

    # Set up loss info
    loss_names = ['VaDE Loss', 'Reconstruction Loss', 'KLD']

    # Set up dataloader for training
    trainset   = Subset(dataset, dataset.trainval_idx)
    dataloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    # Train only Variational Autoencoders
    for param in model.parameters():
        param.requires_grad = True
    for param in model.classifier.parameters():
        param.requires_grad = False

    model.train() # Train mode
    for epoch in range(start_epoch, num_epochs+1):
        epoch_loss = np.zeros_like(loss_names, dtype=float)
        epoch_time = 0.0

        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs))

        # Update coefficients
        [beta] = update_coefficients(epoch, criterion['coefficients'])
        
        # Iterate through batches
        for (i_batch, (features, attributes, labels)) in enumerate(dataloader, start=1):
            start = time.time_ns()

            # Forward pass
            [RL, KLD] = batch_loss(model, features, attributes, labels, criterion['function'])
            
            loss = RL + beta * KLD

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time_ns()

            # Update
            epoch_loss += np.array([loss.item(), RL.item(), KLD.item()])
            epoch_time += end-start

            if i_batch < len(dataloader) and verbose:
                print_progress(i_batch, len(dataloader), end-start, loss.item() / labels.shape[0])
            else:
                epoch_loss /= len(trainset)
                epoch_time /= len(dataloader)

                if verbose:
                    print_progress(i_batch, len(dataloader), epoch_time, epoch_loss[0])

        # Save feedback for this epoch with tensorboard
        if tensorboard_dir != None:
            inspect_epoch(writer, model, epoch, dict(zip(loss_names, epoch_loss)))

        # Update history & save checkpoint - because loss formula updates, we save on every epoch
        model.embeddings_history = np.append(model.embeddings_history, epoch_loss[0])

        checkpoint = {
            'name': model.name,
            'state_dict': model.state_dict(),
            'embeddings_history': model.embeddings_history,
            'classifier_history': model.classifier_history
        }

        torch.save(checkpoint, 'saved/' + model.name + '.pt')

        if verbose:
            print()

    # Create an embedding visualization for train and test splits
    if tensorboard_dir != None:
        add_embeddings(tensorboard_dir, model, dataset)

    model.eval()


def train_classifier(model,
                     optimizer,
                     loss_function,
                     num_epochs=100,
                     batch_size=100,
                     num_seen=200,
                     num_unseen=400,
                     early_stop=4,
                     top_k_acc=1):
    start_epoch = len(model.classifier_history) + 1

    if start_epoch == num_epochs:
        return
    else:
        if verbose:
            print("\033[1mTraining Classifier\033[0m\n")
        else:
            print('\tTraining Classifier for model: ' + model.name)

    if tensorboard_dir != None:
        if tensorboard_dir[-1] != '/':
            tensorboard_dir += '/'
        
        if start_epoch == 1 and os.path.exists(tensorboard_dir + model.name):
            # Remove old classifier files
            for file in os.listdir(tensorboard_dir + model.name):
                if file.endswith('.classifier'):
                    os.remvoe(file)

        writer = SummaryWriter(tensorboard_dir + model.name, filename_suffix='.classifier')

    for epoch in range(start_epoch, num_epochs+1):
        epoch_loss = 0.0

        if verbose:
            print('Epoch {}/{}'.format(epoch, num_epochs))

        # Train
        model.train()
        for (i_batch, (z, labels)) in enumerate(trainloader, start=1):
            start = time.time_ns()

            # Forward pass
            pred = model.classifier(z)
            loss = loss_function(pred, labels.long())

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            end = time.time_ns()

            # Update
            epoch_loss += loss.item()
            if model.verbose:
                print_progress(i_batch, len(trainloader), end-start, loss.item())

        model.clf_history[0] = np.append(model.clf_history[0], epoch_loss / len(trainloader))

        # Test
        model.eval()
        with torch.no_grad():
            # Forward pass for seen classes
            features, _, labels = dataset[dataset.test_seen_idx]
            _, pred = torch.topk(model.forward(features), top_k_acc, dim=1)
            seen_acc = top_k_accuracy(pred, labels)

            # Forward pass for unseen classes
            features, _, labels = dataset[dataset.test_unseen_idx]
            _, pred = torch.topk(model.forward(features), top_k_acc, dim=1)
            unseen_acc = top_k_accuracy(pred, labels)

        # Calculate total accuracy
        if seen_acc == 0 or unseen_acc == 0:
            acc = 0
        else:
            acc = 2*seen_acc*unseen_acc / (seen_acc + unseen_acc)

        if verbose:
            print("Test: S = {:.1f}| U = {:.1f}| \033[1mH = {:.1f}\033[0m \n".format(seen_acc, unseen_acc, acc))

        # Create tensorboard
        if tensorboard_dir != None:
            losses = {'Classifier Loss': epoch_loss, 'S': seen_acc, 'U': unseen_acc, 'H': acc}
            add_to_tensorboard(writer, model, epoch, losses)

        # Update history & save checkpoint iff current epoch is the best
        model.clf_history[1] = np.append(model.clf_history[1], seen_acc)
        model.clf_history[2] = np.append(model.clf_history[2], unseen_acc)
        model.clf_history[3] = np.append(model.clf_history[3], acc)

        if len(model.clf_history[3]) == 1 or acc > max(model.clf_history[3][:-1]):
            checkpoint = {
                'name': model.name,
                'vae_history': model.vae_history,
                'clf_history': model.clf_history,
                'state_dict': model.state_dict()
            }

            torch.save(checkpoint, 'saved/' + model.name + '.pt')

        # Early Stop
        if early_stop != None:
            if len(model.clf_history[3]) > 20 and acc <= min(model.clf_history[3][-early_stop-1:-1]):
                if model.verbose:
                    print('Stopped at epoch {} because H-accuracy stopped improving\n'.format(epoch))
                return np.max(model.clf_history[3])
    
    return np.max(model.clf_history[3])


def main(opt):
    # Init model
    dataset = DatasetGZSL(opt.dataset, opt.device)
    
    if opt.num_shots > 0:
        dataset.transfer_features(opt.num_shots)

    model = CSVaDE(opt.name,
                   device = opt.device,
                   cnn_dim = dataset.cnn_dim,
                   att_dim = dataset.att_dim,
                   embeddings_dim = 64,
                   num_classes = len(dataset.classes),
                   load_pretrained = opt.load_pretrained,
                   reset_classifier = opt.reset_classifier)

    # Train embeddings
    optimizer = optim.Adam(model.parameters(), lr=1e-4, amsgrad=True)

    criterion = {
        'function': nn.L1Loss(reduction='sum'),

        'coefficients': {
            'beta':  {'start': 1,  'stop': 10, 'value': 1},
        }
    }

    train_embeddings(model, dataset, optimizer, criterion)

    # Train classifier
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    
    train_classifier(model, optimizer, nn.NLLLoss(), top_k_acc=k)


if __name__ == '__main__':
    parser = ArgumentParser(description='Trains the CSVaDE model.')

    parser.add_argument('name')

    parser.add_argument('--dataset', '-d', default='AWA1')
    parser.add_argument('--num-shots', '-n', type=int, default=0)

    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--load-pretrained', action='store_true')
    parser.add_argument('--reset-classifier', action='store_true')

    parser.add_argument('--top-k-acc', '-k', type=int, default=1)

    opt = parser.parse_args()

    main(opt)
