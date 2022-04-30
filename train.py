import torch.optim
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model):
        self.epochs = 1
        self.batch_size = 1000
        self.learning_rate = 0.001
        self.model = model
        self.criterion = nn.CrossEntropyLoss()
        return

    def train_validation(self, valid_batch):
        self.model.eval()
        correct = 0
        total = 0
        sum_loss = 0.0
        sum_rmse = 0.0
        for i in range(len(valid_batch[0])):
            text = valid_batch[1][i]
            mark = valid_batch[0][i]
            mark_h = self.model(text)
            loss = self.criterion(mark_h, mark)
            pred = torch.argmax(mark_h)
            correct += (pred == mark).float().sum()
            total += 1
            sum_loss += loss.item()
            sum_rmse += np.sqrt(mean_squared_error([pred], [mark]))
        return sum_loss / total, correct / total, sum_rmse / total

    def train_model(self):
        parameters = filter(lambda x: x.requires_grad, self.model.parameters())
        optimizer = torch.optim.Adam(parameters, lr=self.learning_rate)
        data = self.model.data_loader.load_data()
        plt.figure()
        for i in range(self.epochs):
            train_batch, valid_batch = self.model.data_loader.create_batch(data, self.batch_size)
            sum_loss = 0.0
            total = 0
            total_loss = []
            print("\nEpoch:", i + 1)
            for j in range(len(train_batch[0])):
                self.model.train()
                text = train_batch[1][j]
                mark = train_batch[0][j]
                predicted = self.model(text)
                optimizer.zero_grad()
                loss = self.criterion(predicted, mark)
                loss.backward()
                optimizer.step()
                sum_loss += loss.item()
                total += 1
                if j % 100 == 0:
                    val_loss, val_acc, val_rmse = self.train_validation(valid_batch)
                    total_loss.append(sum_loss / total)
                    print("Train loss %.3f | Value loss %.3f | Value accuracy %.3f | Value RMSE %.3f" % (
                    total_loss[-1], val_loss, val_acc, val_rmse))
            name = "Epoch " + str(i + 1)
            plt.plot(total_loss, label=name)
        plt.legend()
        plt.show()
        return
