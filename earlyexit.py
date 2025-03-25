import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Bloc de base PreActBlock du ResNet
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

# Modèle PreActResNet14 avec Early Exit
class PreActResNet14_EE(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(PreActResNet14_EE, self).__init__()
        self.in_planes = 64
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.layer1 = self._make_layer(block, 32, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 64, num_blocks[1], stride=2)
        
        # Première sortie intermédiaire (Early Exit)
        self.exit1 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes)
        )
        
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
    
    def forward(self, x, threshold=0.9):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        
        # Sortie intermédiaire avec Early Exit
        exit1_logits = self.exit1(out)
        exit1_probs = F.softmax(exit1_logits, dim=1)
        if exit1_probs.max(1)[0].mean().item() > threshold:
            return exit1_logits
        
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Fonction pour créer le modèle PreActResNet14 avec Early Exit
def PreActResNet14_EE_Model():
    return PreActResNet14_EE(PreActBlock, [2, 2, 1, 1])

# Configuration de l'appareil (GPU ou CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialisation du modèle
model = PreActResNet14_EE_Model().to(device)

# Charger le modèle déjà entraîné avec un fichier de checkpoint
checkpoint = torch.load('mybestmodelmain.pth')
model.load_state_dict(checkpoint['net'], strict=False)  # Chargement des poids du modèle dans le réseau

# Charger les données CIFAR-10
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalisation avec la moyenne et l'écart type de CIFAR-10
])

trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)

trainloader = DataLoader(trainset, batch_size=128, shuffle=True)
testloader = DataLoader(testset, batch_size=128, shuffle=False)

# Fonction de perte et optimiseur
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# Fonction pour entraîner le modèle
def train_model(model, trainloader, testloader, device, epochs=15):
    for epoch in range(epochs):
        model.train()  # Mode entraînement
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()  # Rétablir les gradients à zéro
            outputs = model(images)  # Passer les images dans le modèle
            loss = criterion(outputs, labels)  # Calculer la perte
            loss.backward()  # Calculer les gradients
            optimizer.step()  # Mettre à jour les poids
            running_loss += loss.item()
            _, predicted = outputs.max(1)  # Prendre l'étiquette prédite
            correct_train += predicted.eq(labels).sum().item()
            total_train += labels.size(0)
        
        # Calcul de la précision et de la perte moyenne pour cette époque
        train_loss = running_loss / len(trainloader)
        train_accuracy = 100 * correct_train / total_train
        print(f"Epoch {epoch+1}/{epochs}, Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")
        
        # Évaluer le modèle sur le jeu de test
        model.eval()  # Mode évaluation
        correct_test = 0
        total_test = 0
        with torch.no_grad():  # Pas de calcul des gradients pour l'évaluation
            for images, labels in testloader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, predicted = outputs.max(1)
                correct_test += predicted.eq(labels).sum().item()
                total_test += labels.size(0)
        
        test_accuracy = 100 * correct_test / total_test
        print(f"Test Accuracy: {test_accuracy:.2f}%")

# Entraînement du modèle
train_model(model, trainloader, testloader, device, epochs=15)

torch.save({'net': model.state_dict()}, 'mybestmodel_EE.pth')