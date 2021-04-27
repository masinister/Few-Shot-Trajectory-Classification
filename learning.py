import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from tqdm import tqdm
import os
from utils import train_test_split


def train(model, trainloader, testloader, device, epochs):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    # optimizer = optim.SGD(model.parameters(), lr = 5e-4, momentum=0.9)
    max_acc = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        bar = tqdm(enumerate(trainloader, 0), total = len(trainloader))
        for i, (x1, x2, y) in bar:
            X1, X2, labels = Variable(x1), Variable(x2), Variable(y)

            optimizer.zero_grad()

            outputs = model(X1, X2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            bar.set_description('Epoch %d, loss: %.3f, acc: %.3f'%(epoch + 1, running_loss, correct/total))

        correct, total = test(model, testloader, device)
        acc = correct / total
        print("Validation: %d/%d=%.2f Accuracy."%(correct, total, acc))
        if acc > max_acc:
            max_acc = acc
            model.save("testmodel.pth")
    return model

def test(model, dataloader, device):
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for i, (x1, x2, y) in enumerate(dataloader, 0):
            X1, X2, labels = Variable(x1), Variable(x2), Variable(y)
            outputs = model(X1, X2)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    return correct, total
