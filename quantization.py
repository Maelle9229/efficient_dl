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
import pruning
import thop
from thop import profile

# Configuration
config = {
    "learning_rate": 0.001,
    "batch_size": 128,
    "epochs": 1
}

# Normalization for CIFAR10
normalize_scratch = transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))

transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize_scratch,
])

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
net = PreActResNet14()
criterion = nn.CrossEntropyLoss()

# We load the dictionary
loaded_cpt = torch.load('mybestmodelpruned.pth')

net=net.to(device)

rootdir = '/opt/img/effdl-cifar10/'
c10test = CIFAR10(rootdir, train=False, download=True, transform=transform_test)
testloader = DataLoader(c10test, batch_size=config["batch_size"])

# Evaluate on test set
net.eval()
correct_test = 0
total_test = 0
test_loss = 0.0
test_batches = 0
        
with torch.no_grad():
    for images, labels in testloader:
        images, labels = images.to(device), labels.to(device)
        #images, labels = images.to(device).half(), labels.to(device).half()  # Convertit aussi les entrées en float16
        print(f"Images shape: {images.shape}") 

        outputs = net(images)
        loss = criterion(outputs, labels)
        test_loss += loss.item()
        test_batches += 1
        _, predicted = outputs.max(1)
        correct_test += predicted.eq(labels).sum().item()
        total_test += labels.size(0)
        
test_accuracy = 100 * correct_test / total_test
avg_test_loss = test_loss / len(testloader)

#Sauvegarde du modèle avec ses poids et l'hyperparamètre
state = {
    'net': net.state_dict(),  # Les poids du modèle
}


# Calcul du score
dummy_input = torch.randn(1, 3, 32, 32).to(device)  # Batch size = 1
#dummy_input = torch.randn(1, 3, 32, 32).half().to(device)  # Batch size = 1
macs, params = profile(net, inputs=(dummy_input,))

print(f"MACs: {macs:,}")
print(f"Params: {params:,}")

def compute_score(p_s, p_u, q_w, q_a, w, f):
    ref_param = 5.6e6  # Référence pour ResNet18
    ref_ops = 2.8e8    # Référence pour ResNet18

    param_score = ((1 - (p_s + p_u)) * (q_w / 32) * w) / ref_param
    ops_score = ((1 - p_s) * (max(q_w, q_a) / 32) * f) / ref_ops

    return param_score + ops_score

# Exemple d'utilisation :
score = compute_score(
    pruning.p_s, 
    pruning.p_u, 
    q_w=16,
    q_a=16,
    w=params,
    f=macs)
print("Score du modèle:", score)

# Sauvegarder le modèle et l'hyperparamètre dans un fichier
torch.save(state, 'mybestmodelquantizated.pth')