import numpy as np
import torch
import torch.nn.functional as F
import torch as th
import torch.nn as nn
import warnings
from collections import defaultdict
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning
from tqdm import tqdm

def load(emb_file_path):
    emb_dict = {}

    with open(emb_file_path, 'r') as emb_file:
        for i, line in enumerate(emb_file):
            if i == 0:
                train_para = line[:-1]
            else:
                index, emb = line[:-1].split('\t')
                emb_dict[index] = np.array(emb.split()).astype(np.float32)

    return train_para, emb_dict

device = torch.device('cuda:0')
emb_file_path = './PubMed/emb_t5_ablation.dat'
train_para, emb_dict = load(emb_file_path)
label_file_path = './PubMed/label.dat'
label_test_path = './PubMed/label.dat.test'

class MLP_Decoder(nn.Module):
    def __init__(self, hdim, nclass):
        super(MLP_Decoder, self).__init__()
        self.layer1 = nn.Linear(hdim, hdim)
        self.layer2 = nn.Linear(hdim, hdim)
        self.final_layer = nn.Linear(hdim, nclass)
        
        self.skip_weight_1_to_2 = nn.Parameter(torch.rand(1))
        self.skip_weight_1_to_3 = nn.Parameter(torch.rand(1))
        self.skip_weight_2_to_3 = nn.Parameter(torch.rand(1))
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        h1 = self.sigmoid(self.layer1(h))
        
        h2_input = h1 + self.skip_weight_1_to_2 * h
        h2 = self.sigmoid(self.layer2(h2_input))
        
        final_input = h2 + self.skip_weight_1_to_3 * h + self.skip_weight_2_to_3 * h1
        output = self.final_layer(final_input)
        output = self.sigmoid(output)
        
        return output
    
class Disease_MLP(nn.Module):
    def __init__(self, disease_dim, n_class):
        super(Disease_MLP, self).__init__()
        self.decoder = MLP_Decoder(disease_dim, n_class)
    
    def forward(self, disease_emb):
        pred = self.decoder(disease_emb)
        return pred
    
def unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict):
    labels, embeddings = [], []

    for file_path in [label_file_path, label_test_path]:
        with open(file_path, 'r') as label_file:
            for line in label_file:
                index, _, _, label = line[:-1].split('\t')
                labels.append(label)
                embeddings.append(emb_dict[index])
        
    labels, embeddings = np.array(labels).astype(int), np.array(embeddings)

    macro, micro = [], []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=999)

    for train_idx, test_idx in skf.split(embeddings, labels):
        clf = Disease_MLP(2048, 8).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)
        best_ma, best_mi = 0, 0

        for i in range(2000):
            clf.train()
            criterion = nn.BCELoss()
            pred = clf(torch.tensor(embeddings[train_idx]).to(device)).squeeze()

            train_labels = F.one_hot(torch.tensor(labels[train_idx]), num_classes=8)
            train_labels = train_labels.to(device).to(torch.float32)
            loss = criterion(pred, train_labels.to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clf.eval()
            with torch.no_grad():
                preds = clf(torch.tensor(embeddings[test_idx]).to(device))

            ma = f1_score(labels[test_idx], preds.argmax(dim=1).cpu(), average='macro')
            mi = f1_score(labels[test_idx], preds.argmax(dim=1).cpu(), average='micro')

            if ma > best_ma:
                best_ma = ma
                best_mi = mi

            if (i+1) % 100 == 0:
                print("Epoch:", i, "Loss:", round(loss.item(), 4), "Macro F1:", round(best_ma, 4), "Micro F1:", round(best_mi, 4))


        macro.append(best_ma)
        micro.append(best_mi)

    print(macro)
    print(micro)

    return np.mean(macro), np.mean(micro)

score = unsupervised_single_class_single_label(label_file_path, label_test_path, emb_dict)
print(score)