import torch
import dataset



model = torch.load('/home/emile/PythonProjects/DeepLearning_SimilarityFunctions/results/test_cluster.pth', weights_only = False, map_location=torch.device('cpu'))
train_dataloader, dev_dataloader, test_dataloader = dataset.get_dataloaders(path_to_data, batch_size, max_len, class_of_interest=[0,1])

model.eval()
batch_loss = 0
i = 0
preds = []
targets = []
losses = []
batch_embeds = []
for batch, (inputs, batch_targets) in enumerate(test_dataloader):
    inputs, batch_targets = inputs.to(device), batch_targets.to(device)
    batch_preds = model.forward_encode(inputs)
    batch_clf = batch_preds[:,0,:].squeeze().detach().numpy()
    batch_embeds.append(batch_clf)

import matplotlib
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

pca = PCA()
embeds_pca = pca.fit_transform(embeds)
print(embeds_pca.shape)
pca.fit(embeds).explained_variance_ratio_

plt.scatter(embeds_pca[:,0], embeds_pca[:,1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend()

from sklearn.manifold import TSNE
X_embedded = TSNE(n_components=2, learning_rate='auto',
                  init='random', perplexity=3).fit_transform(embeds)
plt.scatter(X_embedded[:,0], X_embedded[:,1], c=labels, cmap=matplotlib.colors.ListedColormap(colors))
plt.legend()
