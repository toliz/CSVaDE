import torch
import numpy as np

from torch.distributions import MultivariateNormal
from torch.nn.functional import cosine_similarity

from models import CSVaDE
from data import DatasetGZSL

from matplotlib import pyplot as plt

model = CSVaDE('csvade', 2048, 85, 50, load_pretrained=True, reset_classifier=False)
dataset = DatasetGZSL('AWA2', 'cpu', purpose='test')

# Calculate theoretical distributions (from attributes)
_, mu, logvar = model.att_encoder(dataset.attributes)
sigma = torch.exp(logvar).sqrt()

dist = [MultivariateNormal(mu[i], torch.diag(sigma[i])) for i in range(50)]

mu    = mu.detach().numpy()
sigma = sigma.detach().numpy()

# Calculate data distributions (from images)
mu_data = [None for i in range(50)]
sigma_data = [None for i in range(50)]

features, attributes, labels = dataset[np.concatenate([dataset.test_seen_idx, dataset.test_unseen_idx])]
for c in range(50):
    idx = np.where(labels == c)[0]
    idx = np.random.choice(idx, 100, replace=True)

    _, points, _ = model.cnn_encoder(features[idx])
    mu_data[c]    = torch.mean(points, dim=0).detach().numpy()
    sigma_data[c] = torch.var(points, dim=0).sqrt().detach().numpy()

# Plot distributions for 3 seen and 3 unseen classes
for c in np.concatenate([dataset.seen_classes[:3], dataset.unseen_classes[:3]]):
    continue
    fig, (ax1, ax2) = plt.subplots(1, 2, num=dataset.classes[c], figsize=(10, 5))
    fig.suptitle(dataset.classes[c])

    ax1.hist(mu_data[c], alpha=0.5, bins=20, color='blue')
    ax1.hist(mu[c]     , alpha=0.5, bins=20, color='red')
    ax1.set_title('mu histogram')
    ax1.legend(['data', 'theoretical'])

    ax2.hist(sigma_data[c], alpha=0.5, bins=20, color='blue')
    ax2.hist(sigma[c]     , alpha=0.5, bins=20, color='red')
    ax2.set_title('sigma histogram')
    ax2.legend(['data', 'theoretical'])

    plt.show()

# Find closest cluster (neighbor) to each cluster
neighbor = [None for i in range(50)]
for i in range(50):
    r = np.linalg.norm(mu[i] - mu, axis=1)
    neighbor[i] = r.argsort()[1]

"""for i in range(50):
    print("{:20}:".format(dataset.classes[i]), end='\t')
    print([np.linalg.norm(mu[i] - mu_data[i]), np.linalg.norm(sigma_data[i])], end='\t')
    print([np.linalg.norm(mu[i] - mu[neighbor[i]]), np.linalg.norm(sigma_data[neighbor[i]])], end='\t')
    print(np.linalg.norm(mu_data[i] - mu[neighbor[i]]), end='\t')
    print(np.min(np.linalg.norm(mu_data[i] - mu, axis=1)), end='\n\n')"""

# Plot class seperation
fig, (ax1, ax2) = plt.subplots(1, 2, num='Data seperation', figsize=(10, 5))
fig.suptitle('Data separation')

sep = []
sep_data = []
for i in dataset.seen_classes:
    j = neighbor[i]
    sep.append(np.linalg.norm(mu[i] - mu[j]) / (np.linalg.norm(sigma[i]) + np.linalg.norm(sigma[j])))
    sep_data.append(np.linalg.norm(mu_data[i] - mu_data[j]) / (np.linalg.norm(sigma_data[i]) + np.linalg.norm(sigma_data[j])))

ax1.hist(sep_data, alpha=0.5, bins='auto', color='blue')
ax1.hist(sep     , alpha=0.5, bins='auto', color='red')
ax1.set_title('Seen classes')
ax1.legend(['data', 'theoretical'])

sep = []
sep_data = []
for i in dataset.unseen_classes:
    j = neighbor[i]
    sep.append(np.linalg.norm(mu[i] - mu[j]) / (np.linalg.norm(sigma[i]) + np.linalg.norm(sigma[j])))
    sep_data.append(np.linalg.norm(mu_data[i] - mu_data[j]) / (np.linalg.norm(sigma_data[i]) + np.linalg.norm(sigma_data[j])))

ax2.hist(sep_data, alpha=0.5, bins='auto', color='blue')
ax2.hist(sep     , alpha=0.5, bins='auto', color='red')
ax2.set_title('Unseen classes')
ax2.legend(['data', 'theoretical'])

plt.show()

# Accuracies based on probabilistic inference
# Forward pass for seen classes
features, _, labels = dataset[dataset.test_seen_idx]
datapoints = model.cnn_encoder(features)[1]
print(np.array([dist[i].log_prob(datapoints).detach().numpy() for i in range(50)]))
exit()
seen_acc = top_k_accuracy(pred, labels)

# Forward pass for unseen classes
features, _, labels = dataset[dataset.test_unseen_idx]
_, pred = torch.topk(model.forward(features), top_k_acc, dim=1)
unseen_acc = top_k_accuracy(pred, labels)

cnn, att, labels = dataset[np.random.choice(dataset.test_unseen_idx, 1000)]
datapoints = model.cnn_encoder(cnn)[1]

acc = 0
for datapoint, label in zip(datapoints, labels):
    probs = np.array([dist[i].log_prob(datapoint).item() for i in range(50)])
    if label.item() == np.argmax(probs):
        acc += 1

print(acc/1000)

features, attributes, labels = dataset[dataset.test_seen_idx]

l = []
for i in range(50):
    j = neighbor[i]
    l.append((torch.norm(mu[i] - mu[j]) / (torch.norm(sigma[i]) + torch.norm(sigma[j]))).item())

n, bins, patches = plt.hist(l, bins='auto', color='#0504aa')
plt.show()
plt.clf()

cos = []
dis = []
for c in dataset.seen_classes[:5]:
    idx = np.where(labels == c)[0]
    idx = np.random.choice(idx, 100, replace=True)

    _, points, logvar = model.cnn_encoder(features[idx])

    cos.append(cosine_similarity(torch.mean(points, dim=0), mu[c], dim=0).item())
    dis.append((torch.norm(torch.mean(points, dim=0) - mu[c]) / torch.norm(mu[c])).item())
    #print(cosine_similarity(torch.var(points, dim=0).sqrt(), sigma[c], dim=0))
    #plt.hist(torch.var(points, dim=0).sqrt().detach().numpy(), bins='auto', color='blue')
    #plt.hist(sigma[c].detach().numpy(), bins='auto', color='red')
    #plt.hist(torch.mean(points, dim=0).sqrt().detach().numpy(), bins='auto', color='blue')
    #plt.hist(mu[c].detach().numpy(), bins='auto', color='red')
    #plt.show()

    plt.hist((torch.norm(points - mu[c], dim=1) / torch.norm(points - mu[neighbor[c]], dim=1)).detach().numpy())
    plt.show()
    plt.clf()
    
    
n, bins, patches = plt.hist(cos, bins=10, color='#0504aa')
plt.show()
n, bins, patches = plt.hist(dis, bins=10, color='#0504aa')
plt.show()