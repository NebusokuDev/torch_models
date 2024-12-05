class CC(nn.Module):
    def __init__(self):
        super(CC, self).__init__()

    def forward(self, pred_map, true_map):
        batch_size = pred_map.size(0)
        correlations = []

        for i in range(batch_size):
            pred = pred_map[i].view(-1)
            true = true_map[i].view(-1)

            # ピアソン相関係数
            correlation = torch.corrcoef(torch.stack([pred, true]))[0, 1]
            correlations.append(correlation.item())

        return torch.tensor(correlations).mean()
