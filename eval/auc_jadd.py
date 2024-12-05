import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

class AUCJudd(nn.Module):
    def __init__(self):
        super(AUCJudd, self).__init__()

    def forward(self, pred_map, true_map):
        batch_size = pred_map.size(0)
        aucs = []

        for i in range(batch_size):
            pred = pred_map[i].view(-1)
            true = true_map[i].view(-1)
            auc = roc_auc_score(true.cpu().numpy(), pred.cpu().numpy())
            aucs.append(auc)

        return torch.tensor(aucs).mean()

class AUCShuffled(nn.Module):
    def __init__(self):
        super(AUCShuffled, self).__init__()

    def forward(self, pred_map, true_map):
        batch_size = pred_map.size(0)
        aucs = []

        for i in range(batch_size):
            pred = pred_map[i].view(-1)
            true = true_map[i].view(-1)
            shuffled_pred = pred[torch.randperm(len(pred))]
            auc = roc_auc_score(true.cpu().numpy(), shuffled_pred.cpu().numpy())
            aucs.append(auc)

        return torch.tensor(aucs).mean()
