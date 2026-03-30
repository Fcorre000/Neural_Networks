# Correa, Fernando
# 1002_053_283
# 2026_02_22
# Assignment_02_01

import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset
import math


class DynamicNet(nn.Module):
    def __init__(self, weight_mats, activation_strs):
        super(DynamicNet, self).__init__()
        self.layers = nn.ModuleList() #container for linear layers
        self.acts = [] #list to store activation functions

        for i, w_mat in enumerate(weight_mats):
            #w_mat shape is (input_dim + 1, output_dim)
            n_in = w_mat.shape[0] - 1
            n_out = w_mat.shape[1]

            #create the standard pytorch linear layer
            layer = nn.Linear(n_in, n_out)

            #manually inject our specific weights and biases
            with torch.no_grad():
                #row 0 is bias, pytorchis bias is shape(n_out)
                layer.bias.copy_(torch.from_numpy(w_mat[0, :]))

                #rows 1 onwards are weights, pytorch weight is shape (n_out, n_in)
                #we transpose our (n_in, n_out) matrix to match
                layer.weight.copy_(torch.from_numpy(w_mat[1:, :].T))

            self.layers.append(layer)

            #map string names to pytorch activation modules
            a_str = activation_strs[i].lower()
            if a_str == 'relu': self.acts.append(nn.ReLU())
            elif a_str == 'sigmoid': self.acts.append(nn.Sigmoid())
            else: self.acts.append(nn.Identity()) # 'linear' activation
    
    def forward(self, x):
        #sequential execution of layers and activation
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.acts[i](x)
        return x



def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0, loss_func='mse', val_split=(0.8, 1.0), seed=7321):

    #-------- DATA PREP AND VALIDATION SPLITTING ----------#
    #calculate the number of samples in the input
    n_samples = x_train.shape[0]

    #determine the start and end indices for the validation set using floor as per instructions
    val_start = int(math.floor(val_split[0] * n_samples))
    val_end = int(math.floor(val_split[1] * n_samples))

    #extract indices for validation and training
    val_indices = list(range(val_start, val_end))
    train_indices = [i for i in range(n_samples) if i not in val_indices]

    #split the numpy arrays using the indices
    x_val_np, y_val_np = x_train[val_indices], y_train[val_indices]
    x_train_np, y_train_np = x_train[train_indices], y_train[train_indices]

    #convert numpy arrays to pytorch float tensors for processing
    x_train_torch = torch.from_numpy(x_train_np).float()
    y_train_torch = torch.from_numpy(y_train_np).float()
    x_val_torch = torch.from_numpy(x_val_np).float()
    y_val_torch = torch.from_numpy(y_val_np).float()

    #-------- WEIGHT INIT(HANDLING BIAS) ----------#
    weight_matrices = []

    #check if 'layers' is a list of ints (architecture) or a list of numpy arrs (pre-set weights)
    if isinstance(layers[0], int) or isinstance(layers[0], np.int64):
        n_in = x_train.shape[1] #input dimension (number of features)
        for nodes in layers:
            #re-seed for every layer as per common acedemic requirements
            np.random.seed(seed)
            #create matrix: rows=input+1 (for bias), cols=number of nodes in this layer
            w = np.random.randn(n_in + 1, nodes).astype(np.float32)
            weight_matrices.append(w)
            n_in = nodes #update input dimension for the next layer
    else:
        #use the provided weight matrices directly
        weight_matrices = [w.astype(np.float32) for w in layers]

    model = DynamicNet(weight_matrices, activations)

    # ----- TRAINING AND CUSTOM SVM LOSS ------
    #select the loss function
    l_func = loss_func.lower()
    if l_func == 'mse':
        criterion = nn.MSELoss()
    elif l_func == 'crossentropy':
        criterion = nn.CrossEntropyLoss()
    elif l_func == 'svm':
        def svm_loss(outputs, targets):
            margins = torch.clamp(1.0 - targets * outputs, min=0)
            return torch.mean(margins)
        criterion = svm_loss
    
    #use the standard SGD optimizer
    optimizer = SGD(model.parameters(), lr=alpha)

    # ------ TRAINING LOOP AND MAE CALC ------#
    error_list = []
    if epochs > 0:
        #create a dataloader for minibatch processing (no shuffling as per rules)
        dataset = TensorDataset(x_train_torch, y_train_torch)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        for epoch in range(epochs):
            model.train() # set to training mode
            for b_x, b_y in loader:
                optimizer.zero_grad() #clear old gradients
                preds = model(b_x) #forward pass
                loss = criterion(preds, b_y) #compute loss
                loss.backward() #backward pass(compute gradients)
                optimizer.step() #update weights
            
            #post eopch validation: calc MAE
            model.eval() # set to evaluation mode (freezes weights)
            with torch.no_grad():
                val_preds = model(x_val_torch)
                #compute MAE: mean(|targets - predictions|)
                mae = torch.mean(torch.abs(y_val_torch - val_preds)).item()
                error_list.append(mae)
        
    #final outputs
    model.eval()
    with torch.no_grad():
        if len(val_indices) > 0:
            #pass the entire validation set through the model
            final_val_out = model(x_val_torch).numpy()
        else:
            #handle empty validation set case
            final_val_out = np.array([]).reshape(0, y_train.shape[1])
    

    #extract weights back to numpy format (bias as row 0)
    final_weight_matrices = []
    for layer in model.layers:
        #pytorch stores weights as (out_features, in_features)
        # we need (in_features, out_features) for the return format
        w_numpy = layer.weight.detach().numpy().T

        #pytorch stores bias as 1D arr of (out_features)
        #we reshape it to (1, out_features) to make it the first row
        b_numpy = layer.bias.detach().numpy().reshape(1, -1)

        #stack them: bias on top, weights below
        combined = np.vstack([b_numpy, w_numpy])
        final_weight_matrices.append(combined)
    
    #final return statement
    return [final_weight_matrices, error_list, final_val_out]
