import argparse
import torch
from torch.nn import CrossEntropyLoss
from torch.optim import Adam
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder
from models.nanonet import Nanonet
from trainer import Trainer

# コマンドライン引数の定義
parser = argparse.ArgumentParser(description='Nanonet Training')
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr', type=float, default=1e-3)
parser.add_argument('--weight-decay', type=float, default=5e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--gpu', type=int, default=0)  # 0または1の整数に変更
parser.add_argument('--train-data', type=str)
parser.add_argument('--val-data', type=str)
parser.add_argument('--test-data', type=str)
parser.add_argument('--save-path', type=str, default='./trained_models')
parser.add_argument('--classes', type=int, default=1000)

if __name__ == '__main__':
    args = parser.parse_args()

    # モデルの設定
    model = Nanonet(classes=args.classes)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = CrossEntropyLoss()

    # デバイスの設定
    device = torch.device('cpu')
    if args.gpu and torch.cuda.is_available():
        device = torch.device('cuda')

    # データ前処理の設定
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # データローダーの設定
    train_dataloader = DataLoader(ImageFolder(args.train_data, transform=transform), batch_size=args.batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(ImageFolder(args.test_data, transform=transform), batch_size=args.batch_size,
                                 shuffle=True)

    # トレーナーのインスタンス化と訓練開始
    trainer = Trainer(train_dataloader, test_dataloader, criterion, device, save_path=args.save_path)
    trainer.fit(model, optimizer, epochs=args.epochs)
