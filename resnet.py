#%% Kütüphanelerin Yüklenmesi

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from PIL import Image
import torchvision
import torch.nn as nn
import cv2
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch
import os
import time
import random
from tqdm import tqdm
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from torchvision import datasets, transforms, models
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data.sampler import SubsetRandomSampler
#%% Veri Seti Dosya İşlemleri

inmevar = "C:/Users/kadir/Desktop/kadirnar/Yapay Zeka\Deep Leaarning\Classification/dataset/inme/inmevar/"
inmeyok = "C:/Users/kadir/Desktop/kadirnar/Yapay Zeka\Deep Leaarning\Classification/dataset/inme/inmeyok/"
data_dir = "C:/Users/kadir/Desktop/kadirnar/Yapay Zeka\Deep Leaarning\Classification/dataset/inme/"

#%% İnme Var Görüntüleri Görüntüleme

inmevar_files      = [os.path.join(inmevar , x) for x in os.listdir(inmevar)]
inmevar_images    =  [cv2.imread(x) for x in random.sample(inmevar_files, 2)]

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(inmevar_images):
    plt.subplot(len(inmevar_images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#%% İnme Yok Görüntüleri Görüntüleme

inmeyok_files      = [os.path.join(inmeyok , x) for x in os.listdir(inmeyok)]
inmeyok_images    =  [cv2.imread(x) for x in random.sample(inmeyok_files, 2)]

plt.figure(figsize=(20,10))
columns = 5
for i, image in enumerate(inmeyok_images):
    plt.subplot(len(inmeyok_images) / columns + 1, columns, i + 1)
    plt.imshow(image)

#%% Cuda Aktif Etme

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

#%%
def load_split_train_test(datadir, valid_size = .2):
    
    train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                           transforms.RandomResizedCrop(224),
                                           transforms.RandomHorizontalFlip(),
                                           transforms.ToTensor(),
                                           transforms.Normalize([0.485, 0.456, 0.406],
                                                                [0.229, 0.224, 0.225])
    ])
    
    test_transforms = transforms.Compose([transforms.Resize(255),
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
                   sampler=train_sampler, batch_size=64)
    
    testloader = torch.utils.data.DataLoader(test_data,
                   sampler=test_sampler, batch_size=64)
    
    return trainloader, testloader

trainloader, testloader = load_split_train_test(data_dir, .2)
print(trainloader.dataset.classes)
['InmeVar', 'InmeYok']

#%%

model = models.resnet50(pretrained=True)
print(model)


#%%

for param in model.parameters():
    param.requires_grad = False
    
model.fc = nn.Sequential(nn.Linear(2048, 512),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(512, 2),
                                 nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.SGD(model.fc.parameters(), lr=0.001)
model.to(device)

#%%
epochs = 3
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device),labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    test_loss += batch_loss.item()
                    
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals =  top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
            train_losses.append(running_loss/len(trainloader))
            test_losses.append(test_loss/len(testloader))                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
torch.save(model, 'covidlmodel.pth')



#%%

plt.plot(train_losses, label='Training loss')
plt.plot(test_losses, label='Validation loss')
plt.legend(frameon=False)
plt.show()


#%%


