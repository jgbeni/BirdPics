import numpy as np
import torch
from torch.utils.data import  DataLoader
import torchvision.models as models
import torch.nn as nn
import utils.data_preprocessing as dp
from tqdm import tqdm

class load_model:
    def __init__(self, model_name='resnet50', num_classes=3, pretrained=True):
        self.model_name = model_name
        self.num_classes = num_classes
        self.pretrained = pretrained
        self.weight_path = 'models/' + model_name + '_retrained.pth'

        def get_model():
            if self.model_name == 'resnet50':
                model = models.resnet50()
                num_ftrs = model.fc.in_features
                model.fc = nn.Linear(num_ftrs, self.num_classes)
                if self.pretrained:
                    model.load_state_dict(torch.load(self.weight_path,map_location=torch.device('cpu')))
                    model.eval()
            elif self.model_name == 'vgg16':
                model = models.vgg16()
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)
                if self.pretrained:
                    model.load_state_dict(torch.load(self.weight_path,map_location=torch.device('cpu')))
                    model.eval()
            elif self.model_name == 'vgg19':
                model = models.vgg19()
                num_ftrs = model.classifier[-1].in_features
                model.classifier[-1] = nn.Linear(num_ftrs, self.num_classes)
                if self.pretrained:
                    model.load_state_dict(torch.load(self.weight_path,map_location=torch.device('cpu')))
                    model.eval()
            else:
                raise ValueError(f"Model {self.model_name} not supported in this version.")
            
            return model
        self.get_model = get_model

def create_dataloader(X, Y, batch_size=64, shuffle=False, num_workers=2, train=False):
    Y_enc = dp.prepare_labels(Y)
    dataset = dp.HDF5Dataset(X, Y_enc, train=train)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    return dataloader

def eval_model(model, X_test, Y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testloader = create_dataloader(X_test, Y_test)

    model.to(device)

    correct = 0
    total = 0
    classes = ['swallow','swift','martin']
    n_class_correct = [0 for i in range(len(classes))]
    n_class_samples = [0 for i in range(len(classes))]
    label_vec,pred_vec = [],[]
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            images = images.to(device)
            labels = labels.to(device).long()
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            for i in range(list(labels.shape)[0]):
                label_vec.append(labels[i].cpu().item())
                pred_vec.append(predicted[i].cpu().item())
                label = labels[i]
                pred = predicted[i]
                if (label == pred):
                    n_class_correct[label] += 1
                n_class_samples[label] += 1
    accuracy = 100 * correct / total
    print(f'Overall Accuracy of the model on the test images: {accuracy} %')
    class_acc = []
    for i in range(len(classes)):
        class_acc.append(100.0 * n_class_correct[i] / n_class_samples[i])
        print(f'Accuracy of {classes[i]}: {class_acc[i]} %')
    return accuracy, class_acc, np.array(label_vec, int), np.array(pred_vec, int)


