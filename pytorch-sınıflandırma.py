# Kütüphanelerin Yüklenmesi

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import numpy as np
import time
from collections import OrderedDict
import os
from torch.utils.data import SubsetRandomSampler

#%%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Veri setinin train ve test olarak ayırma

datadir = "dataset/"  

def load_split_train_test(datadir, valid_size = .2):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(p=1),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([transforms.Resize(256),
                                          transforms.CenterCrop(224),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])
    ])
    train_data = datasets.ImageFolder(datadir,       
                    transform=train_transforms)
    
    test_data = datasets.ImageFolder(datadir,
                    transform=test_transforms)
    
    num_train = len(train_data)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))
    np.random.shuffle(indices)
    
    train_idx, test_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    test_sampler = SubsetRandomSampler(test_idx)
    
    trainloader = torch.utils.data.DataLoader(train_data,
                   sampler=train_sampler, batch_size=128)
    
    validloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=128)
    
    return trainloader, validloader

trainloader, validloader = load_split_train_test(datadir, .2)
print(trainloader.dataset.classes)

# Efficientnet modelini kurma

# efficientnet model 
model = models.efficientnet.efficientnet_b7(pretrained=True)
for param in model.parameters():
    param.requires_grad = False

print(model)

# FC Katmanı

model = models.efficientnet.efficientnet_b2(pretrained=True)
for param in model.parameters():
    param.requires_grad = False
print(model)


classifier = nn.Sequential(OrderedDict([
    ('fc1', nn.Linear(1408, 512)),
    ('relu', nn.LeakyReLU()),
    ('Lazy_norm', nn.LazyBatchNorm1d(512)),
    ('fsec1', nn.Flatten()),
    ('fc2', nn.Linear(512, 2)),
    
    ('output', nn.LogSoftmax(dim=1))
]))

model.classifier = classifier
model.to(device)

# Optimizer
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.classifier.parameters(), lr=0.01)



# Validation

def validation(model, criterion, validloader):
    valid_loss = 0
    accuracy = 0
    for data in validloader:
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        outputs = model(images)
        loss = criterion(outputs, labels)
        valid_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        accuracy += (predicted == labels).sum().item()
    return valid_loss, accuracy


# Modeli Eğitme

def train_model(model, criterion, optimizer, epochs=50, print_every=40, steps=0, trainloader=trainloader, validloader=validloader):
    start = time.time()
    steps = 0
    running_loss = 0
    print_every = 40
    for e in range(epochs):
        model.train()
        for ii, (inputs, labels) in enumerate(trainloader):
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs.to(device))
            loss = criterion(outputs, labels.to(device))
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                model.eval()
                with torch.no_grad():
                    valid_loss, accuracy = validation(model, criterion, validloader)
                print("Epoch: {}/{}.. ".format(e+1, epochs),
                      "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                      "Validation Loss: {:.3f}.. ".format(valid_loss/len(validloader)),
                      "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                running_loss = 0
                model.train()
    end = time.time()
    print("Training time: {:.3f}".format(end-start))
    return model
train_model(model, criterion, optimizer, epochs=50, print_every=40, steps=0, trainloader=trainloader, validloader=validloader)
