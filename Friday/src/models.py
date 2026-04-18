import torch
import torch.nn as nn
from torchvision import models

def build_feature_extractor(num_classes):
    """
    Sub-step 2: Feature Extraction using Pre-trained ResNet18
    Backbone is frozen. Only head is trained.
    """
    # Load pre-trained ResNet18
    # Using 'pretrained=True' is deprecated but standard for backwards compatibility unless we use weights parameter.
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet18(pretrained=True)
        
    # Freeze the backbone
    for param in model.parameters():
        param.requires_grad = False
        
    # Replace the classification head
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def build_fine_tuning_model(num_classes):
    """
    Sub-step 3: Fine-tuning ResNet18
    Unfreeze parts (or all) of the network. We'll unfreeze layer4 and the fc layer.
    """
    try:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    except:
        model = models.resnet18(pretrained=True)
        
    # Freeze all layers initially
    for param in model.parameters():
        param.requires_grad = False
        
    # Unfreeze layer4
    for param in model.layer4.parameters():
        param.requires_grad = True
        
    # Replace classification head (automatically requires_grad=True)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def build_from_scratch(num_classes):
    """
    Sub-step 6: Random Initialization
    Training from scratch with no pre-trained weights.
    """
    try:
        model = models.resnet18(weights=None)
    except:
        model = models.resnet18(pretrained=False)
        
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    return model

def train_model(model, dataloaders, criterion, optimizer, num_epochs=3, device='cpu'):
    """
    Generalized training loop capable of handling the dataset configurations
    """
    model = model.to(device)
    
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
                
            running_loss = 0.0
            running_corrects = 0
            
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                
                optimizer.zero_grad()
                
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
                        
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                
    return model, best_acc
