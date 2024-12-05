from torch.nn import Module


class SegVit(Module):
    def __init__(self):
        super().__init__()

        self.features = 