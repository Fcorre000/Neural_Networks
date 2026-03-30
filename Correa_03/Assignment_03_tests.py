import os
import random
import numpy as np
import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import confusion_matrix
# Modify the line below based on your last name
from Correa_03_01 import train_cnn_torch, confusion_matrix


def seed_all(manual_seed):
    random.seed(manual_seed)
    np.random.seed(manual_seed)
    torch.manual_seed(manual_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(manual_seed)


def get_random_data_loader(batch_size):
    x_train = 2*np.random.rand(100, 1, 28, 28) - 1
    y_train = np.random.randint(low=0, high=10, size=(100,))
    x_test = 2*np.random.rand(100, 1, 28, 28) - 1
    y_test = np.random.randint(low=0, high=10, size=(100,))

    # Get validation split
    val_dataset = TensorDataset(torch.tensor(x_test, dtype=torch.float32), torch.tensor(y_test, dtype=torch.long))
    val_data_loader = DataLoader(dataset=val_dataset, batch_size=len(x_test), shuffle=False)
    # Get train split
    train_dataset = TensorDataset(torch.tensor(x_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.long))
    train_data_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=False)
    return train_data_loader, val_data_loader


def get_mnist_loader(batch_size):
    transform = transforms.Compose([
        transforms.PILToTensor(),
        transforms.Lambda(lambda x: x/255.0 - 0.5),
    ])
    train_ds = MNIST(root='../data', train=True, download=True, transform=transform)
    test_ds = MNIST(root='../data', train=False, download=True, transform=transform)
    train_data_loader = DataLoader(dataset=train_ds, batch_size=batch_size, shuffle=True)
    test_data_loader = DataLoader(dataset=test_ds, batch_size=batch_size, shuffle=True)
    return train_data_loader, test_data_loader


def test_model_architecture():
    seed_all(manual_seed=7321)
    train_dl, test_dl = get_random_data_loader(batch_size=32)
    model, loss_history, con_mat, accuracy_history = train_cnn_torch(train_dl=train_dl, test_dl=test_dl, lr=0.01, epochs=0, test_mode=False)

    layers = [a for a in model.net.named_children()]
    assert len(layers) == 15
    # First Conv layer, ReLU
    assert isinstance(layers[0][1], torch.nn.modules.Conv2d)
    assert (layers[0][1].in_channels == 1) and (layers[0][1].out_channels == 8) 
    assert (layers[0][1].kernel_size == (3, 3)) and (layers[0][1].stride == (1, 1)) and (layers[0][1].padding == 'same')
    assert isinstance(layers[1][1], torch.nn.modules.activation.ReLU)
    # Second Conv layer, ReLU, MaxPool
    assert isinstance(layers[2][1], torch.nn.modules.Conv2d)
    assert (layers[2][1].in_channels == 8) and (layers[2][1].out_channels == 16)
    assert (layers[2][1].kernel_size == (3, 3)) and (layers[2][1].stride == (1, 1)) and (layers[2][1].padding == 'same')
    assert isinstance(layers[3][1], torch.nn.modules.activation.ReLU)
    assert isinstance(layers[4][1], torch.nn.modules.pooling.MaxPool2d)
    assert (layers[4][1].kernel_size == (2, 2) or layers[4][1].stride == 2)
    # Third Conv layer
    assert isinstance(layers[5][1], torch.nn.modules.Conv2d)
    assert (layers[5][1].in_channels == 16) and (layers[5][1].out_channels == 32)
    assert (layers[5][1].kernel_size == (3, 3)) and (layers[5][1].stride == (1, 1)) and (layers[5][1].padding == 'same')
    assert isinstance(layers[6][1], torch.nn.modules.activation.ReLU)
    # Fourth Conv layer
    assert isinstance(layers[7][1], torch.nn.modules.Conv2d)
    assert (layers[7][1].in_channels == 32) and (layers[7][1].out_channels == 64)
    assert (layers[7][1].kernel_size == (3, 3)) and (layers[7][1].stride == (1, 1)) and (layers[7][1].padding == 'same')
    assert isinstance(layers[8][1], torch.nn.modules.activation.ReLU)
    assert isinstance(layers[9][1], torch.nn.modules.pooling.MaxPool2d)
    assert (layers[9][1].kernel_size == (2, 2) or layers[9][1].stride == 2)
    # Flatten followed by linear layers
    assert isinstance(layers[10][1], torch.nn.modules.flatten.Flatten)
    assert isinstance(layers[11][1], torch.nn.modules.linear.Linear)
    assert (layers[11][1].in_features == 3136) and (layers[11][1].out_features == 512)
    assert isinstance(layers[12][1], torch.nn.modules.activation.ReLU)
    assert isinstance(layers[13][1], torch.nn.modules.linear.Linear)
    assert (layers[13][1].in_features == 512) and (layers[13][1].out_features == 10)
    assert isinstance(layers[14][1], torch.nn.modules.activation.Softmax) and (layers[14][1].dim == -1)
    

def test_model_output():
    seed_all(manual_seed=7321)
    train_dl, test_dl = get_random_data_loader(batch_size=32)
    model, loss_history, con_mat, accuracy_history = train_cnn_torch(train_dl=train_dl, test_dl=test_dl, lr=0.01, epochs=0, test_mode=False)
    assert (isinstance(loss_history, list) or isinstance(loss_history, np.ndarray)) and len(loss_history) == 0
    assert (isinstance(accuracy_history, list) or isinstance(accuracy_history, np.ndarray)) and len(accuracy_history) == 0
    assert (con_mat is None)

    if os.path.exists('confusion_matrix.png'):
        os.remove('confusion_matrix.png')
    seed_all(manual_seed=7321)
    train_dl, test_dl = get_random_data_loader(batch_size=32)
    model, loss_history, con_mat, accuracy_history = train_cnn_torch(train_dl=train_dl, test_dl=test_dl, lr=0.01, epochs=4, test_mode=False)
    reltol=1e-03
    assert np.allclose(loss_history, np.array([2.303292, 2.3127532, 2.3325627, 2.344552]),reltol) or np.allclose(loss_history, [2.303292, 2.3127532, 2.3325627, 2.344552],reltol)
    assert os.path.exists('confusion_matrix.png')


def test_save_model():
    if os.path.exists('cnn.pt'):
        os.remove('cnn.pt')
    seed_all(manual_seed=7321)
    train_dl, test_dl = get_random_data_loader(batch_size=32)
    model, loss_history, con_mat, accuracy_history = train_cnn_torch(train_dl=train_dl, test_dl=test_dl, lr=0.01, epochs=0, test_mode=False)
    assert os.path.exists('cnn.pt')
    try:
        model.load_state_dict(torch.load('cnn.pt'))
        os.remove('cnn.pt')
    except RuntimeError:
        raise RuntimeError('Unable to load saved model')


def test_confusion_matrix():
    y_true = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    y_pred = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    con_mat = confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert con_mat.shape == (10, 10)
    gt = confusion_matrix(y_true=y_true, y_pred=y_pred)
    assert np.all(con_mat == gt), f"Your confusion matrix is incorrect.  It should be:\n{gt} not \n{con_mat}"


def test_accuracy_on_mnist():
    seed_all(manual_seed=7321)
    train_dl, test_dl = get_mnist_loader(batch_size=128)
    model, loss_history, con_mat, accuracy_history = train_cnn_torch(train_dl=train_dl, test_dl=test_dl, lr=0.001, epochs=10, test_mode=True)

    assert (isinstance(accuracy_history, list) or isinstance(accuracy_history, np.ndarray)) and len(accuracy_history) == 10
    assert np.allclose(accuracy_history, np.array([0.078125,  0.0703125, 0.0703125, 0.1796875, 0.2109375, 0.2734375, 0.3671875, 0.296875,  0.359375,  0.3046875]))


