import time

import torch
import torch.nn as nn
from matplotlib import pyplot
from torch.autograd import Variable
from torch.optim import Adam
from torch.utils.data import DataLoader

from constants import BATCH_SIZE, LEARNING_RATE, LEARNING_RATE_DECAY, EPOCHS, LOSS_IMPROVEMENT, STAGNATING_EPOCHS
from dataset import ImageClassifierDataset

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Unit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Unit, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_channels, kernel_size=3, out_channels=out_channels, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.activation_function = nn.Tanh()

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        output = self.activation_function(output)

        return output


class SimpleNet(torch.nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()

        self.net = nn.Sequential(
            self.unit(3, 32),
            self.maxPool(16),
            self.unit(32, 64),
            self.maxPool(8),
            self.unit(64, 128),
            self.avgPool(2)
        )

        self.fc = nn.Linear(in_features=128, out_features=1)

    @staticmethod
    def unit(in_channels, out_channels):
        return Unit(in_channels=in_channels, out_channels=out_channels)

    @staticmethod
    def maxPool(kernel_size):
        return nn.MaxPool2d(kernel_size=kernel_size)

    @staticmethod
    def avgPool(kernel_size):
        return nn.AvgPool2d(kernel_size=kernel_size)

    def forward(self, input):
        output = self.net(input)
        output = output.view(-1, 128)
        output = self.fc(output)
        output = torch.sigmoid(output)
        return output


def adjust_learning_rate():
    for param_group in optimizer.param_groups:
        param_group["lr"] = LEARNING_RATE * LEARNING_RATE_DECAY


def save_models(epoch):
    torch.save(model.state_dict(), f"models/network_epoch_{epoch}")
    print("Checkpoint saved")


def test():
    model.eval()
    test_accuracy = 0.0
    test_loss = 0.0
    for i, (images, labels) in enumerate(test_loader):
        if cuda_avail:
            images = Variable(images.cuda())
            labels = Variable(labels.cuda())
        outputs = model(images)
        prediction = torch.round(outputs.data)
        loss = loss_fn(outputs, labels)
        test_loss += loss.cpu().data.item() * images.size(0)
        test_accuracy += torch.sum(torch.eq(prediction, labels.data))

    test_accuracy = test_accuracy / test_set_size
    test_loss = test_loss / test_set_size

    return test_accuracy, test_loss


def train(epochs):
    best_loss = 1
    epochs_without_improvement = 0
    losses = []
    print("Training Started...")
    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_acc = 0.0
        train_loss = 0.0
        for i, (images, labels) in enumerate(train_loader):
            if cuda_avail:
                images = Variable(images.cuda())
                labels = Variable(labels.cuda())

            optimizer.zero_grad()
            outputs = model(images)

            loss = loss_fn(outputs, labels)

            loss.backward()

            optimizer.step()

            train_loss += loss.cpu().data.item() * images.size(0)
            prediction = torch.round(outputs.data)
            train_acc += torch.sum(torch.eq(prediction, labels.data))

        adjust_learning_rate()

        train_acc /= train_set_size
        train_loss /= train_set_size

        test_acc, test_loss = test()

        if test_loss + LOSS_IMPROVEMENT < best_loss:
            if epoch != 0:
                save_models(epoch)
            best_loss = test_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

            if epochs_without_improvement == STAGNATING_EPOCHS:
                print(
                    f"Epoch {epoch}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc}, TestLoss: {test_loss} Time: {time.time() - start_time}")

                print(f"No improvement in {epochs_without_improvement} epochs, stopping...")
                losses.append(test_loss)
                break

        print(
            f"Epoch {epoch}, Train Accuracy: {train_acc} , TrainLoss: {train_loss} , Test Accuracy: {test_acc}, TestLoss: {test_loss} Time: {time.time() - start_time}")
        losses.append(test_loss)
    pyplot.plot(losses)
    pyplot.show()


if __name__ == '__main__':
    # create the dataset
    dataset = ImageClassifierDataset()
    dataset.load_data()
    train_set, test_set = dataset.split()
    train_set_size = len(train_set)
    test_set_size = len(test_set)

    # Create a loader for the training set and testing set
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    # Check if gpu support is available
    cuda_avail = torch.cuda.is_available()

    # Create model, optimizer and loss function
    model = SimpleNet()
    if cuda_avail:
        model.cuda()
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=0.0000)
    loss_fn = nn.BCELoss()
    train(EPOCHS)
