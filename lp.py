import numpy as np
import torch
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import warnings
from collections import defaultdict
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import UndefinedMetricWarning, ConvergenceWarning

op1, op2, op3 = [], [], []
entity_count, relation_count = 0, 0

with open('./PubMed/node.dat', 'r') as original_meta_file:
    for line in original_meta_file:
        temp1, temp2, temp3 = line.split('\t')
        op1.append(temp1)
        op2.append(temp2)
        op3.append(temp3)

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
emb_file_path = './PubMed/emb_llama2_ablation.dat'
train_para, emb_dict = load(emb_file_path)

link_test_file = './PubMed/sample.dat'
pos_edges, neg_edges = [], []

with open(link_test_file, 'r') as original_meta_file:
    for line in original_meta_file:
        temp = []
        temp1, temp2, temp3 = line[:-1].split('\t')

        temp.append(int(temp1))
        temp.append(int(temp2))
        if temp3 == '1': pos_edges.append(temp)
        if temp3 == '0': neg_edges.append(temp)

pos_edges_tensor = torch.tensor(pos_edges, dtype=torch.long)
neg_edges_tensor = torch.tensor(neg_edges, dtype=torch.long)

node_emb = []
d2l = dict()
line = 0

for key, values in emb_dict.items():
    node_emb.append(values)
    d2l[key] = line
    line += 1

node_emb = np.array(node_emb)
node_emb = torch.Tensor(node_emb)

class LMNN(nn.Module):
    def __init__(self, node_emb, pos_edges_tensor, neg_edges_tensor):
        super(LMNN, self).__init__()
        self.embeddings = nn.Embedding.from_pretrained(node_emb, freeze=False)
        self.margin = 50
        self.pos_edges_tensor = pos_edges_tensor
        self.neg_edges_tensor = neg_edges_tensor

    def decode(self, h_all, idx):
        h = h_all

        emb_node1 = h[idx[:, 0]]
        emb_node2 = h[idx[:, 1]]

        sqdist = emb_node1 * emb_node2
        sqdist = torch.sum(sqdist, dim=1)
        return sqdist
    
    def forward(self):
        pos_scores = self.decode(self.embeddings.weight, self.pos_edges_tensor)
        neg_scores = self.decode(self.embeddings.weight, self.neg_edges_tensor)

        loss = neg_scores - pos_scores + self.margin
        loss[loss < 0] = 0
        loss = torch.sum(loss)
        return loss
    
model = LMNN(node_emb, pos_edges_tensor, neg_edges_tensor).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
for epoch in range(1000):
    model.train()
    optimizer.zero_grad()

    train_loss = model()
    loss = train_loss

    loss.backward()
    optimizer.step()

emb_trained = model.embeddings.weight
emb_trained_list = emb_trained.cpu().detach().numpy()

with open('./PubMed/sample_emb_t5.dat', 'w') as file:
    file.write('pubmed41\n')

    for i in range(len(op2)):
        file.write(f'{i}\t')
        file.write(' '.join(emb_trained_list[i].astype(str)))
        file.write('\n')

emb_file_path = './PubMed/sample_emb_t5.dat'
train_para, emb_dict = load(emb_file_path)

class MLP_Decoder(nn.Module):
    def __init__(self, hdim, nclass):
        super(MLP_Decoder, self).__init__()
        self.layer1 = nn.Linear(hdim, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 128)
        self.layer4 = nn.Linear(128, 128)
        self.final_layer = nn.Linear(128, nclass)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.skip_weight_1_to_2 = nn.Parameter(torch.rand(1))
        self.skip_weight_1_to_3 = nn.Parameter(torch.rand(1))
        self.skip_weight_1_to_4 = nn.Parameter(torch.rand(1))
        self.skip_weight_2_to_3 = nn.Parameter(torch.rand(1))
        self.skip_weight_2_to_4 = nn.Parameter(torch.rand(1))
        self.skip_weight_3_to_4 = nn.Parameter(torch.rand(1))

    def forward(self, h):
        h1 = self.relu(self.layer1(h))
        h2 = self.relu(self.layer2(h1 + self.skip_weight_1_to_2 * h1))
        h3 = self.relu(self.layer3(h2 + self.skip_weight_2_to_3 * h2 + self.skip_weight_1_to_3 * h1))
        h4 = self.relu(self.layer4(h3 + self.skip_weight_3_to_4 * h3 + self.skip_weight_2_to_4 * h2 + self.skip_weight_1_to_4 * h1))
        output = self.sigmoid(self.final_layer(h4))
        
        return output

class Disease_MLP(nn.Module):
    def __init__(self, disease_dim, n_class):
        super(Disease_MLP, self).__init__()
        self.decoder = MLP_Decoder(disease_dim, n_class)
    
    def forward(self, disease_emb):
        pred = self.decoder(disease_emb)
        return pred

def mrrc(pred, label):
    n = len(pred)
    sorted_idx = np.argsort(pred)[::-1]
    ranks = np.zeros(n)

    for i in range(n):
        ranks[sorted_idx[i]] = i+1

    pos = np.where(label == 1)[0]
    if len(pos) == 0:
        return 0
    
    pos_rank = ranks[pos]
    return 1.0 / pos_rank

def lp_evaluate(test_file_path, emb_dict):
    posi, nega = defaultdict(set), defaultdict(set)

    with open(test_file_path, 'r') as test_file:
        for line in test_file:
            left, right, label = line[:-1].split('\t')

            if label == '1':
                posi[left].add(right)
            elif label == '0':
                nega[left].add(right)

    edge_embs, edge_labels = defaultdict(list), defaultdict(list)
    for left, rights in posi.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left] * emb_dict[right])
            edge_labels[left].append(1)
    for left, rights in nega.items():
        for right in rights:
            edge_embs[left].append(emb_dict[left] * emb_dict[right])
            edge_labels[left].append(0)

    for node in edge_embs:
        edge_embs[node] = np.array(edge_embs[node])
        edge_labels[node] = np.array(edge_labels[node])

    auc, mrr = cross_validation(edge_embs, edge_labels)
    return auc, mrr

def cross_validation(edge_embs, edge_labels):
    auc, mrr = [], []
    seed_nodes, num_nodes = np.array(list(edge_embs.keys())), len(edge_embs)
    seed = 1
    skf = KFold(n_splits=5, shuffle=True, random_state=seed)

    for fold, (train_idx, test_idx) in enumerate(skf.split(np.zeros((num_nodes, 1)), np.zeros(num_nodes))):
        print(f'Start Evaluation Fold {fold}!')
        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = [], [], [], []

        for each in train_idx:
            train_edge_embs.append(edge_embs[seed_nodes[each]])
            train_edge_labels.append(edge_labels[seed_nodes[each]])
        for each in test_idx:
            test_edge_embs.append(edge_embs[seed_nodes[each]])
            test_edge_labels.append(edge_labels[seed_nodes[each]])

        train_edge_embs, test_edge_embs, train_edge_labels, test_edge_labels = np.concatenate(train_edge_embs), np.concatenate(test_edge_embs), np.concatenate(train_edge_labels), np.concatenate(test_edge_labels)

        best_auc, best_mrr = 0, 0
        clf = Disease_MLP(4096, 1).to(device)
        optimizer = torch.optim.Adam(clf.parameters(), lr=0.001)

        for i in range(1000):
            clf.train()
            criterion = nn.BCELoss()
            pred = clf(torch.tensor(train_edge_embs).to(device)).squeeze()

            loss = criterion(pred, torch.tensor(train_edge_labels).to(device).to(torch.float32))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            clf.eval()
            with torch.no_grad():
                preds = clf(torch.tensor(test_edge_embs).to(device))

            auc1 = roc_auc_score(test_edge_labels, preds.cpu())

            confidence = preds.squeeze().cpu()
            curr_mrr, conf_num = [], 0
            for each in test_idx:
                test_edge_conf = np.argsort(-confidence[conf_num:conf_num+len(edge_labels[seed_nodes[each]])])
                rank = np.empty_like(test_edge_conf)
                curr_mrr.append(1/(1+np.min(rank[np.argwhere(edge_labels[seed_nodes[each]] == 1).flatten()])))
                conf_num += len(rank)

            mrr1 = np.mean(curr_mrr)
            if auc1 > best_auc:
                best_auc = auc1
                best_mrr = mrr1

            if (i+1) % 50 == 0:
                print("Epoch:", i, "Loss:", loss.item(), "AUC:", best_auc, "MRR:", best_mrr)
            
        auc.append(best_auc)
        mrr.append(best_mrr)

    return np.mean(auc), np.mean(mrr)
    
link_test_file = './PubMed/link.dat.test'
link_test_path = f'{link_test_file}'
scores = lp_evaluate(link_test_path, emb_dict)
print(scores)