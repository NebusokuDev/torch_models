from os import path, makedirs
import torch
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter


class Trainer:
    def __init__(self, train_dataloader: DataLoader, test_dataloader: DataLoader, criterion: Module,
                 device: torch.device, show_stride=10, save_path="./trained_models/"):
        self.criterion = criterion
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.show_stride = show_stride
        self.device = device
        self.save_path = save_path
        self.writer = SummaryWriter(log_dir='./logs')

    def _training_step(self, model: Module, optimizer: Optimizer, epoch: int):
        model.train()
        total_loss = 0
        for batch_index, (images, labels) in enumerate(self.train_dataloader):
            images, labels = images.to(self.device), labels.to(self.device)
            optimizer.zero_grad()
            predict = model(images)
            loss = self.criterion(predict, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if batch_index % self.show_stride == 0:
                print(f"Batch {batch_index}/{len(self.train_dataloader)} - Loss: {loss.item():.5f}")

        avg_loss = total_loss / len(self.train_dataloader)
        self.writer.add_scalar('Loss/train', avg_loss, epoch)  # TensorBoardに書き込み
        print(f"Training Loss: {avg_loss:.5f}")
        return avg_loss

    def _testing_step(self, model: Module):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in self.test_dataloader:
                images, labels = images.to(self.device), labels.to(self.device)
                predicted = model(images)
                loss = self.criterion(predicted, labels)
                total_loss += loss.item()

                # 精度計算
                _, predicted_labels = torch.max(predicted, 1)
                correct += (predicted_labels == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(self.test_dataloader)
        accuracy = correct / total
        self.writer.add_scalar('Loss/test', avg_loss)  # TensorBoardに書き込み
        self.writer.add_scalar('Accuracy/test', accuracy, 0)  # TensorBoardに書き込み
        print(f"Test Loss: {avg_loss:.5f}, Accuracy: {accuracy * 100:.2f}%")
        return avg_loss, accuracy

    def fit(self, model: Module, optimizer: Optimizer, epochs: int = 100):
        best_loss = float("inf")
        best_accuracy = 0.0
        no_improvement = 0
        patience = 10  # 早期終了のためのパラメータ

        for epoch in range(epochs):
            print(f"Epoch {epoch + 1}/{epochs}")
            train_loss = self._training_step(model, optimizer, epoch)
            test_loss, accuracy = self._testing_step(model)
            print("-" * 50)

            is_best_loss = best_loss > test_loss
            is_best_accuracy = best_accuracy < accuracy

            if is_best_loss:
                best_loss = test_loss
                no_improvement = 0
            else:
                no_improvement += 1

            if is_best_accuracy:
                best_accuracy = accuracy
                no_improvement = 0
            else:
                no_improvement += 1

            # 最良モデルを保存
            if is_best_loss or is_best_accuracy:
                model_path = f"{self.save_path}/best_model.pth"
                if not path.exists(self.save_path):
                    makedirs(self.save_path)
                torch.save(model.state_dict(), model_path)

            # 早期終了
            if no_improvement >= patience:
                print("Early stopping: No improvement in the last few epochs")
                break

    def __call__(self, model: Module, optimizer: Optimizer, epochs: int = 100):
        self.fit(model, optimizer, epochs)
