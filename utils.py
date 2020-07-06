import os
import sys
import json
import torch
import numpy as np

from copy import deepcopy

from torch.optim import Adam
from torch.nn import L1Loss, NLLLoss


def init_weights(module):
    if type(module) == torch.nn.Linear:
        torch.nn.init.constant_(module.bias, 0)
        torch.nn.init.xavier_uniform_(module.weight, gain=0.5)


def print_progress(current, total, time, loss, length = 60):
    """
    Call in a loop to create terminal progress bar
    """
    time = int(time / 1000000)
    filledLength = int(length * current // total)
    bar = '█' * filledLength + '-' * (length - filledLength)
    percent = '{:.1f}'.format(100 * (current / float(total)))
    
    print('{}/{} |{}| {}% - {:d} ms/batch - loss: {:.2f}'.format(
        current, total, bar, percent, time, loss), end = "\r")

    # Print New Line on Complete
    if current == total: 
        print()


def update_coefficients(epoch, coeffecient_dict):
    coefficients = []

    for (coefficient, update) in coeffecient_dict.items():
        coefficient = min(max((epoch - update['start']) / (update['stop'] - update['start']), 0), 1) * update['value']
        coefficients += [coefficient]

    return coefficients


def inspect_epoch(writer, model, epoch, losses):
    # Track all trainable parameters
    for (name, param) in model.named_parameters():
        if param.requires_grad:
            writer.add_histogram(name, param, global_step=epoch)

    # Track losses & their coefficients (if any)
    for (name, v) in losses.items():
        writer.add_scalar(name, v, global_step=epoch)


def add_embeddings(dir, model, dataset):
    for idx_type in ['trainval', 'test_seen', 'test_unseen']:
        idx = getattr(dataset, idx_type + '_idx')
        cnn, att, labels = dataset[np.random.choice(idx, 1000)]

        datapoints = torch.cat([model.cnn_encoder(cnn)[1], model.att_encoder(att)[1]])
        modalities = ['cnn'] * len(labels) + ['att'] * len(labels)
        labels     = [dataset.classes[label] for label in labels] * 2 # switch from integer to string labelling

        # Create embeddings folder
        if not os.path.exists('{}{}/embeddings/{}'.format(dir, model.name, idx_type)):
            os.makedirs('{}{}/embeddings/{}'.format(dir, model.name, idx_type))
    
        # Create tensors
        with open('{}{}/embeddings/{}/tensors.tsv'.format(dir, model.name, idx_type), 'w') as file:
            for datapoint in datapoints:
                file.write('\t'.join([str(x.item()) for x in datapoint]) + '\n')

        # Create metadata
        with open('{}{}/embeddings/{}/metadata.tsv'.format(dir, model.name, idx_type), 'w') as file:
            file.write('Label\tModality\tAll\n')
            for (label, modality) in zip(labels, modalities):
                file.write('{}\t{}\t{} | {}\n'.format(label, modality, label, modality))

        # Create embeddings indexing
        with open('{}{}/projector_config.pbtxt'.format(dir, model.name), 'a') as file:
            file.write("""embeddings {
                        tensor_name: "embeddings:<idx_type>"
                        tensor_path: "embeddings/<idx_type>/tensors.tsv"
                        metadata_path: "embeddings/<idx_type>/metadata.tsv"}\n""".replace('<idx_type>', idx_type))


def top_1_accuracy(pred, true):
    if torch.is_tensor(pred):
        pred = pred.tolist()
    if torch.is_tensor(true):
        true = true.tolist()

    accuracies = dict.fromkeys(true, 0) # correct samples per class
    total = dict.fromkeys(true, 0)      # total samples per class

    for (p, t) in zip(pred, true):
        accuracies[t] += (p == t)
        total[t] += 1

    return 100 * np.average(np.array(list(accuracies.values())) / np.array(list(total.values())))


def top_k_accuracy(pred, true):
    if torch.is_tensor(pred):
        pred = pred.tolist()
    if torch.is_tensor(true):
        true = true.tolist()

    accuracies = dict.fromkeys(true, 0) # correct samples per class
    total = dict.fromkeys(true, 0)      # total samples per class

    for (p, t) in zip(pred, true):
        accuracies[t] += (t in p)
        total[t] += 1

    return 100 * np.average(np.array(list(accuracies.values())) / np.array(list(total.values())))


def str_to_class(classname):
    """
    Converts a string to a python class
    """
    return getattr(sys.modules[__name__], classname)


def get_model_name(config):
    name = config['general']['dataset']

    # General
    if config['general']['num_shots'] > 0:
        name += '_{}-shots'.format(config['general']['num_shots'])

    # Architecture
    if config['architecture']['embeddings_dim'] != 200:
        name += '_d=' + str(config['architecture']['embeddings_dim'])
    if config['architecture']['cnn_hidden_layers'] != [2000]:
        name += '_cnn_hidden=' + str(config['architecture']['cnn_hidden_layers'])
    if config['architecture']['att_hidden_layers'] != [2000]:
        name += '_att_hidden=' + str(config['architecture']['att_hidden_layers'])

    # Embeddings
    if 'optimizer' in config['embeddings']:
        if config['embeddings']['optimizer']['name'] != 'Adam':
            name += '_optim=' + config['embeddings']['optimizer']['name']
        else:
            name += '_lr=' + str(config['embeddings']['optimizer']['settings']['lr'])
            if config['embeddings']['optimizer']['settings']['amsgrad'] == True:
                name += '_amsgrad'

    # Classifier
    if config['classifier']['num_seen'] != 100:
        name += '_num_seen=' + str(config['classifier']['num_seen'])
    if config['classifier']['num_unseen'] != 400:
        name += '_num_unseen=' + str(config['classifier']['num_unseen'])

    return name


def _unroll_dict(d):
    """
    Helper function for create configs.

    If a parameter is a dictionary form, it searches recurrsively for list in the dictionary values.
    """
    unrolled = [dict.fromkeys(d)]

    for (param, value) in d.items():
        if type(value) is dict:
            value = _unroll_dict(value)
        
        if type(value) is list:
            new_unrolled = []

            for v in value:
                for existing in unrolled:
                    existing_ = deepcopy(existing)
                    existing_[param] = eval(v) if (type(v) is str and v.startswith('[')) else v
                    new_unrolled.append(existing_)

            unrolled = new_unrolled
        
        else:
            for i in range(len(unrolled)):
                unrolled[i][param] = eval(value) if (type(value) is str and value.startswith('[')) else value

    
    return unrolled


def create_configs(filename):
    """
    Loads a JSON file containing one or multiple configurations and returns a list with all the possible configurations.
    If the JSON file contains a single configuration, it returns a list with a single element: this configuration.

    The functon searches for lists in the parameters and for each one it replicates the current configuration as many
    times as the elements of the list; then appends each configuration with the corresponding parameter. This procedure 
    continues recurrsively till all possible conbinations are reached at the end of the JSON file.

    If you wish to parse a list as a single parameter then quote it (e.g. lr = "[1e-3, 2e-3]")
    """
    with open(filename) as file:
        initial = json.load(file)

    configs = [{'general': {}, 'architecture': {}, 'embeddings': {}, 'classifier': {}}]

    for part in ['general', 'architecture', 'embeddings', 'classifier']:
        for (param, value) in initial[part].items():
            # Unrolls dicts with lists in their values
            if type(value) is dict:
                value = _unroll_dict(value)

            # Test multiple value for the parameter
            if type(value) is list:
                new_configs = []
                
                for v in value:
                    for config in configs:
                        config_ = deepcopy(config)
                        config_[part][param] = eval(v) if (type(v) is str and v.startswith('[')) else v
                        new_configs.append(config_)

                configs = new_configs
            # Test single v for the parameter
            else:
                for i in range(len(configs)):
                    configs[i][part][param] = eval(value) if (type(value) is str and value.startswith('[')) else value

    return configs
