class NSS(nn.Module):
    def __init__(self):
        super(NSS, self).__init__()

    def forward(self, pred_map, fixations):
        batch_size = pred_map.size(0)
        nss_scores = []

        for i in range(batch_size):
            pred = pred_map[i].view(-1)
            fixation = fixations[i].view(-1)

            fixation_values = pred[fixation == 1]

            # 正規化
            if fixation_values.numel() > 0:  # 注目領域が存在する場合
                mean = fixation_values.mean()
                std = fixation_values.std()
                nss_scores.append((fixation_values - mean) / std if std != 0 else 0)

        return torch.tensor(nss_scores).mean()
