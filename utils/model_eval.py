import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.models as models
import torch.nn as nn
import utils.data_preprocessing as dp
from sklearn.metrics import confusion_matrix
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

# Create dataloader for evaluation

# Evaluate the model on test data
def eval_model(model, X_test, Y_test):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    testloader = dp.create_dataloader(X_test, Y_test)

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

# Class to load and plot loss and accuracy curves
class loss_acc_loader:
    def __init__(self, model_name,model_dir ='models/'):
        self.model_name = model_name
        self.model_dir = model_dir
        self.model_path = self.model_dir + self.model_name + '_retrained.npz'
        self.data = np.load(self.model_path)
        self.train_loss = self.data['train_loss']
        self.val_loss = self.data['val_loss']
        self.train_acc = self.data['train_acc']
        self.val_acc = self.data['val_acc']
        self.epochs = len(self.train_loss)
        self.epoch_range = range(1, self.epochs + 1)

    # Plot loss curves
    def plot_loss(self):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.figure(figsize=(7, 4))
        plt.plot(self.epoch_range, self.train_loss, label='Train Loss')
        plt.plot(self.epoch_range, self.val_loss, label='Validation Loss')
        plt.title(f'Loss Curves for {self.model_name}', fontsize=18)
        plt.xlabel('Epochs', fontsize=14)
        plt.xticks(self.epoch_range)
        plt.ylabel('Loss', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()

    # Accuracy plot
    def plot_accuracy(self):
        plt.rcParams['font.family'] = 'sans-serif'
        plt.figure(figsize=(7, 4))
        plt.plot(self.epoch_range, self.train_acc, label='Train Accuracy')
        plt.plot(self.epoch_range, self.val_acc, label='Validation Accuracy')
        plt.title(f'Accuracy Curves for {self.model_name}', fontsize=18)
        plt.xlabel('Epochs', fontsize=14)
        plt.xticks(self.epoch_range)
        plt.ylabel('Accuracy', fontsize=14)
        plt.legend()
        plt.tight_layout()
        plt.show()
    
    # Combined plot for loss and accuracy
    def plot_loss_acc(self):
        plt.rcParams['font.family'] = 'sans-serif'
        fig, axes = plt.subplots(2, 1, figsize=(6, 6))

        # Loss plot
        ax1 = axes[0]
        ax1.set_xlabel('Epochs', fontsize=14)
        ax1.set_ylabel('Loss', fontsize=14)
        ax1.plot(self.epoch_range, self.train_loss, label='Train Loss', color='blue', linestyle='-')
        ax1.plot(self.epoch_range, self.val_loss, label='Validation Loss', color='blue', linestyle='--')
        ax1.legend()
        ax1.set_xticks(self.epoch_range)

        # Accuracy plot
        ax2 = axes[1]
        ax2.set_xlabel('Epochs', fontsize=14)
        ax2.set_ylabel('Accuracy', fontsize=14)
        ax2.plot(self.epoch_range, self.train_acc, label='Train Accuracy', color='orange', linestyle='-')
        ax2.plot(self.epoch_range, self.val_acc, label='Validation Accuracy', color='orange', linestyle='--')
        ax2.legend()
        ax2.set_xticks(self.epoch_range)
        ax2.set_ylim(40, 100)
        

        fig.suptitle(f'Loss and Accuracy Curves for {self.model_name}', fontsize=18)
        plt.tight_layout()
        plt.show()
        
def plot_confusion_matrix(labels,predictions,classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function computes and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    
    cm = confusion_matrix(labels, predictions)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], fmt),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()