import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# 量子化モジュール（Vector Quantization）
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim):
        super(VectorQuantizer, self).__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        # コードブックの作成
        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim)

    def forward(self, x):
        # x: [batch_size, channels, height, width]
        # それぞれの特徴ベクトルに最も近いコードベクトルを見つける
        flat_input = x.view(-1, self.embedding_dim)
        distances = (flat_input ** 2).sum(dim=1, keepdim=True) + \
                    (self.embedding.weight ** 2).sum(dim=1) - 2 * torch.matmul(flat_input, self.embedding.weight.t())
        indices = torch.argmin(distances, dim=1).unsqueeze(1)
        quantized = self.embedding(indices)
        quantized = quantized.view_as(x)

        # 量子化後の特徴量を返す
        return quantized, indices


# エンコーダ
class Encoder(nn.Module):
    def __init__(self, in_channels, hidden_channels, embedding_dim):
        super(Encoder, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(hidden_channels, embedding_dim, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.conv3(x)
        return x


# デコーダ
class Decoder(nn.Module):
    def __init__(self, embedding_dim, hidden_channels, out_channels):
        super(Decoder, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(embedding_dim, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(hidden_channels, hidden_channels, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(hidden_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = torch.sigmoid(self.deconv3(x))  # 出力を[0, 1]に収束させるためsigmoidを使用
        return x


# VQ-VAEモデル
class VQVAE(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=3, embedding_dim=64, num_embeddings=512):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(in_channels, hidden_channels, embedding_dim)
        self.vector_quantizer = VectorQuantizer(num_embeddings, embedding_dim)
        self.decoder = Decoder(embedding_dim, hidden_channels, in_channels)

    def forward(self, x):
        z_e = self.encoder(x)
        z_q, indices = self.vector_quantizer(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, indices, z_q


if __name__ == '__main__':
    device = torch.device("gpu" if torch.cuda.is_available() else "cpu")
    model = VQVAE()
    summary(model, (3, 64, 64))