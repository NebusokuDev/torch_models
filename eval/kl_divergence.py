import torch.nn.functional as F

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, pred_map, true_map):
        batch_size = pred_map.size(0)
        kl_divs = []

        for i in range(batch_size):
            pred = pred_map[i] + 1e-6  # 小さな定数を加える
            true = true_map[i] + 1e-6  # 小さな定数を加える

            # 正規化
            pred = pred / pred.sum()
            true = true / true.sum()

            kl_div = torch.sum(true * torch.log(true / pred))
            kl_divs.append(kl_div.item())

        return torch.tensor(kl_divs).mean()
