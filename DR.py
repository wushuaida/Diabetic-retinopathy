


import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision


class DR(Dataset):
    """Diabetic retina dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable object, optional): Optional transform to be applied
                on a sample.
        """
        self.labels = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = os.path.join(self.labels.iloc[idx, 0]+'.png')
        image = io.imread(img_name)
        label = self.labels.iloc[idx, 1]
        sample = {'image': image, 'label': label}
        
        if self.transform:
            sample['image'] = self.transform(sample['image'])
        return sample['image'], sample['label']


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 20, 5, 1)
        self.conv2 = nn.Conv2d(20, 50, 5, 1)
        self.fc1 = nn.Linear(186050, 500)
        self.fc2 = nn.Linear(500, 5)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2, 2)
        x = x.view(-1, 186050)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(train_loader, model, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()       #clean the gradient of param in model 
        output = model(data)        
        loss = F.cross_entropy(output, target)      
        loss.backward()             #do backward, get the gradient
        optimizer.step()            #update the param in model
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()
        
        print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.1f}%'.format(
            epoch, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(),  100. * correct / len(data)))
            

def test(test_loader, model):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            
            data, target = data.to(device), target.to(device)

            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def projected_gradient_descent(model, x, y, loss_fn, num_steps, step_size, eps, clamp=(0,1)):
    """
    Performs the projected gradient descent attack on a batch of images.
    
    Input:
        model: a nn model
        x: batch of data, with size: N*C*H*W
        y: label, with size: N
        loss_fn: a loss function. example: F.nll_loss
        num_steps: int
        step_size: float, to perform x += step_size*grad
        eps: a float in (0,1), the eps-bound
        clamp: make sure the adv is an image
    
    Output:
        x_adv: a transformed image
    
    """
    
    # use detach method to cut off the connection, and make it a leaf.
    x_adv = x.clone().detach().requires_grad_(True).to(x.device)

    for i in range(num_steps):
        _x_adv = x_adv.clone().detach().requires_grad_(True)

        prediction = model(_x_adv)
        loss = loss_fn(prediction, y)
        loss.backward()

        # a required grad leaf can't be operated due to pytorch rule
        with torch.no_grad():
            # Force the gradient step to be a fixed size in a certain norm
            gradients = _x_adv.grad.sign() * step_size
            x_adv += gradients

        # Workaround as PyTorch doesn't have elementwise clip
        x_adv = torch.max(torch.min(x_adv, x + eps), x - eps)
            
        x_adv = x_adv.clamp(*clamp)

    return x_adv.detach()


def pgd_train(train_loader, model, optimizer, epoch, eps):
    """
    using pgd_train method train the model

    """
    
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        
        # generate the adversarial batch w.r.t current model
        data_adv = projected_gradient_descent(model, data, target, loss_fn = F.nll_loss, num_steps = 50, step_size = 0.01, eps = eps)
        
        optimizer.zero_grad()       #clean the gradient of param in model 
        
        output = model(data_adv)        
        loss = F.nll_loss(output, target)     
        
        loss.backward()             #do backward, get the gradient
        optimizer.step()            #update the param in model
        
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct = pred.eq(target.view_as(pred)).sum().item()

        print('Train Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.1f}%'.format(
            epoch, epochs, batch_idx * len(data), len(train_loader.dataset),
            100. * batch_idx / len(train_loader), loss.item(),  100. * correct / len(data)))
        
            
def pgd_test(test_loader, model, eps):    
    model.eval()
    test_loss = 0
    correct = 0
    
    for data, target in test_loader:
        data, target = data.to(device), target.to(device)
        
        data_adv = projected_gradient_descent(model, data, target, loss_fn = F.nll_loss, num_steps = 50, step_size = 0.01, eps = eps)

        output = model(data_adv)
        test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))    
    
if __name__ == "__main__":
    label = pd.read_csv('train.csv')
    adversarial_training = True
    eps = 0.05
    epochs = 10

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
   # device = torch.device('cpu')
    
    Resize = transforms.Resize((256,256))
    ToPIL = transforms.ToPILImage()
    ToTensor = transforms.ToTensor()
    tran = transforms.Compose([
         ToPIL,
         Resize,
         ToTensor
         ])
    
    dataset = DR(csv_file='train.csv', root_dir='', transform=tran)

    batch_size = 16
    validation_split = .2
    shuffle_dataset = True
    random_seed= 42
    
    # Creating data indices for training and validation splits:
    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(validation_split * dataset_size))
    if shuffle_dataset :
        np.random.seed(random_seed)
        np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)    #2930
    valid_sampler = SubsetRandomSampler(val_indices)      #732
    
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, 
                                               sampler=train_sampler)
    validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                    sampler=valid_sampler)
    test_loader = torch.utils.data.DataLoader(dataset, batch_size=16)
    

    model = torchvision.models.resnet34(num_classes=5, pretrained=False)
    model = model.to(device)
    
    
    optimizer = optim.SGD(model.parameters(), lr=0.05)
    
    if adversarial_training:
        for epoch in range(epochs):
            pgd_train(train_loader, model, optimizer, epoch, eps)
            pgd_test(validation_loader, model, eps)
    
    else :
        for epoch in range(epochs):
            train(train_loader, model, optimizer, epoch)
            test(validation_loader, model)
            
            
            
            
            
            
            
            
            
            

    
    
    