import wandb
from torchvision.datasets import CIFAR10
import numpy as np
import torchvision.transforms as transforms
import torch
from torch.utils.data.dataloader import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
import sys
import os
import torchinfo
from torchinfo import summary


# PreActResNet Definition
class PreActBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(PreActBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False)
            )

    def forward(self, x):
        out = F.relu(self.bn1(x))
        shortcut = self.shortcut(out) if hasattr(self, 'shortcut') else x
        out = self.conv1(out)
        out = self.conv2(F.relu(self.bn2(out)))
        out += shortcut
        return out


class PreActResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet, self).__init__()
        self.in_planes = 64
        self.dropout = nn.Dropout(0.3)

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 128, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 256, num_blocks[3], stride=2)
        self.linear = nn.Linear(256 * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)  # Appliquer Dropout ici
        out = self.linear(out)
        return out


def PreActResNet14():
    return PreActResNet(PreActBlock, [2, 2, 1, 1])

# Training Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
net = PreActResNet14().to(device)

# We load the dictionary
loaded_cpt = torch.load('mybestmodel_EE.pth')


# Finally we can load the state_dict in order to load the trained parameters
net.load_state_dict(loaded_cpt['net'], strict=False)


# Configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 64,
    "epochs": 25
}

wandb.init(project="cifar10-tutorial", config=config, 
           name=f"lr_{config['learning_rate']}_bs_{config['batch_size']}_ep_{config['epochs']}")

# Normalization for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

# Data augmentation for training
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize_scratch,
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

# Load CIFAR10 dataset
rootdir = '/opt/img/effdl-cifar10/'
c10train = CIFAR10(rootdir, train=True, download=True, transform=transform_train)
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)

trainloader = DataLoader(c10train, batch_size=config["batch_size"], shuffle=True)
testloader = DataLoader(c10test, batch_size=config["batch_size"])


# Pruning
def apply_pruning(model, p_s, p_u):
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):  # Appliquer uniquement aux Conv2d
            prune.ln_structured(module, name="weight", amount=p_s, dim=1, n=2)  # Pruning structuré (L2 norm)
            prune.l1_unstructured(module, name="weight", amount=p_u)  # Pruning non structuré
            print(f"Pruning applied to {name}")
# Définir les taux de pruning
p_s = 0.1  # 10% de pruning structuré
p_u = 0.4  # 40% de pruning non structuré
apply_pruning(net, p_s, p_u) # Appliquer la pruning à toutes les couches


#BOUCLE entrainement
def train_model(net, trainloader, testloader, device, epochs=config["epochs"]):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    
    for epoch in range(epochs):
        net.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = net(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
        
        train_accuracy = 100 * correct_train / total_train
        train_loss = running_loss / len(trainloader)


        # Evaluate on test set
        net.eval()
        correct_test = 0
        total_test = 0
        test_loss = 0.0
        test_batches = 0
        
        with torch.no_grad():
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = net(images)
                loss = criterion(outputs, labels)
                test_loss += loss.item()
                test_batches += 1
                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)
        
        test_accuracy = 100 * correct_test / total_test
        avg_test_loss = test_loss / len(testloader)
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, Test Loss: {avg_test_loss:.4f}, Test Acc: {test_accuracy:.2f}%")
        wandb.log({"train_loss": train_loss, "train_accuracy": train_accuracy, "test_loss": avg_test_loss, "test_accuracy":test_accuracy})

        avg_test_loss = test_loss / test_batches
        print(f"Test Loss after epoch {epoch+1}: {test_loss / test_batches:.4f}")

train_model(net, trainloader, testloader, device, epochs=config["epochs"])

print("Training completed!")


#Sauvegarde du modèle avec ses poids et l'hyperparamètre 
state = {
    'net': net.state_dict(),  # Les poids du modèle
}

# Sauvegarder le modèle et l'hyperparamètre dans un fichier
torch.save(state, 'mybestmodelpruned.pth')