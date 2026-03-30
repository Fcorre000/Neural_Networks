# Correa, Fernando
# 1000_123_456
# 2026_03_15
# Assignment_03_01

import random
import numpy as np
import torch
from torch import nn
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            #block 1 - 2 conv layers
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #block 2 - 2 more conv layers
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding='same'),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),

            #flatten + fully connected
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
            nn.Softmax(dim=-1) 
        )
    
    def forward(self, x):
        return self.net(x)


def confusion_matrix(y_true, y_pred, n_classes=10):
    """
    Compute the confusion matrix for a multi-class classification task.

    Parameters
    ----------
    y_true : np.ndarray
        Ground-truth class labels (shape: [number_of_samples]).
        Each entry must be an integer class index in {0, ..., n_classes-1}.
    y_pred : np.ndarray
        Predicted outputs from the model.
        This may be:
            - Class indices of shape [number_of_samples], or
            - Class probabilities/logits of shape [number_of_samples, n_classes].
        If probabilities/logits are provided, the predicted class is defined
        as argmax along axis=1 (highest probability).
    n_classes : int, default=10
        Total number of classes.

    Returns
    -------
    cm : np.ndarray
        Confusion matrix of shape (n_classes, n_classes),
        where entry (i, j) represents the number of samples
        whose true label is i and were predicted as class j.

    Notes
    -----
    - Do NOT use external libraries such as sklearn.metrics.confusion_matrix
      or tensorflow.math.confusion_matrix.
    - The confusion matrix is constructed manually using indexing/counting.
    """
    if y_pred.ndim == 2:
        y_pred = np.argmax(y_pred, axis=1)
    
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[int(t)][int(p)] += 1
    return cm




def train_cnn_torch(train_dl, test_dl, lr=0.01, epochs=1, test_mode=False):
    """
    Train a convolutional neural network using PyTorch.

    Parameters
    ----------
    train_dl : torch.utils.data.DataLoader
        DataLoader for the training set (see the provided test file for how it is constructed).
    test_dl : torch.utils.data.DataLoader
        DataLoader for the test set.
    lr : float, default=0.01
        Learning rate for the Adam optimizer.
    epochs : int, default=1
        Number of training epochs. If epochs == 0, do not train; only initialize and return the model.
    debug_mode : bool, default=False
        If True, for each epoch train and evaluate using only the first batch from each DataLoader.
        If False, train and evaluate using the entire dataset.

    Model Architecture (must match exactly; hardcoding is allowed)
    ------------------------------------------------------------
    1) Conv2d: in_channels=?, out_channels=8,  kernel_size=3x3, stride=1x1, padding='same'
    2) ReLU
    3) Conv2d: out_channels=16, kernel_size=3x3, stride=1x1, padding='same'
    4) ReLU
    5) MaxPool2d: kernel_size=2, stride=2
    6) Conv2d: out_channels=32, kernel_size=3x3, stride=1x1, padding='same'
    7) ReLU
    8) Conv2d: out_channels=64, kernel_size=3x3, stride=1x1, padding='same'
    9) ReLU
    10) MaxPool2d: kernel_size=2, stride=2
    11) Flatten
    12) Linear: 512 units
    13) ReLU
    14) Linear: 10 units
    15) Softmax

    Training / Evaluation Requirements
    ----------------------------------
    - Optimizer: Adam with learning rate `lr` (other parameters left default).
    - Loss: categorical cross-entropy (use torch.nn.CrossEntropyLoss).
    - Test loss should be computed over the entire test set (except in debug_mode, use only first test batch).
    - Compute the confusion matrix on the test set and return it as a NumPy array.
    - Plot the confusion matrix using matplotlib `matshow` and save to 'confusion_matrix.png'.
    - Save the PyTorch model to 'cnn.pt' (do not submit this file; it will be checked during testing).
    - Program must run without user interaction (do not require closing figures, etc.).
    - Accuracy must be computed as: accuracy = num_correct / num_total_samples.

    Returns
    -------
    list
        [0] model:
            Trained model (epochs > 0) or initialized model (epochs == 0)
        [1] test_loss_history: np.ndarray or empty
            Test loss history across epochs; if epochs == 0 return empty list/array
        [2] confusion_matrix: np.ndarray or None
            Confusion matrix over the test set; if epochs == 0 return None
        [3] test_accuracy_history: np.ndarray or empty
            Test accuracy history across epochs; if epochs == 0 return empty list/array
    """
    #1. Create Model, optimizer, and loss func
    model = CNN()
    optimizer = Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()

    #2. always save the model (even if epochs =0)
    torch.save(model.state_dict(), 'cnn.pt')

    #handle epochs=0 early exit
    if epochs == 0:
        return [model, [], None, []]

    #3. set up history trackers
    test_loss_history = []
    test_accuracy_history = []

    for epoch in range(epochs):

        #--Training Phase---#
        model.train()
        for images, labels in train_dl:

            #get output from first 14 layers(before softmax)
            logits = model(images)
            
            #compute loss using logits (not softmax output)
            loss = loss_fn(logits, labels)

            #backprop, update the weights
            optimizer.zero_grad() #clear old grad
            loss.backward() #compute new gradients
            optimizer.step() #apply them
            
            #if test_mode only do one batch per epoch
            if test_mode:
                break
        
        #--Evaluation Phase--#
        model.eval()
        total_loss = 0
        num_correct = 0
        num_total = 0
        all_labels = []
        all_predictions = []

        num_batches = 0

        with torch.no_grad(): #turn off gradient tracking (saves memory)
            for images, labels in test_dl:

                output = model(images)    # forward pass 1 only
                predictions = model.net[-1](logits) # just applies softmax to existing result
                
                #accumulate loss
                total_loss += loss_fn(output, labels).item()
                num_batches += 1

                #get predicted class (index of highest probability)
                predicted_classes = torch.argmax(output, dim=1)

                #accumulate correct count
                num_correct += (predicted_classes == labels).sum().item()
                num_total += labels.size(0)

                #store for confusion matrix later
                all_labels.extend(labels.numpy())
                all_predictions.extend(predicted_classes.numpy())

                if test_mode:
                    break
            
        #record this epoch's loss and accuracy
        test_loss_history.append(total_loss / num_batches)
        test_accuracy_history.append(num_correct/ num_total)
    
    #--After all Epochs--#

    #build confusion matrix from collected labels
    cm = confusion_matrix(np.array(all_labels), np.array(all_predictions))

    #plot and save confusion matrix
    
    fig, ax = plt.subplots()
    ax.matshow(cm)
    plt.savefig('confusion_matrix.png')
    plt.close()

    #save the trained model
    torch.save(model.state_dict(), 'cnn.pt')

    #return everything
    return [model, test_loss_history, cm, test_accuracy_history]

    pass
