import os
import sys
sys.path.append(os.getcwd())

from tft_numba_ltp import tft, tft_rc, sigma
import numpy as np
import matplotlib.pyplot as plt
from utils import *
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from timetools import timefn


# RC system RNN based on LTP.
# seed
seed = 70
# savefile path
save_path = 'ltp\\result'
# dataset path
data_path = 'data'
# network settings
num_epochs = 1

# noise standard deviation
sigma = 0.15 # [0,0.5)
#-----------------------------------------------------------#
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cpu = torch.device("cpu")
# print(device)

torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
np.random.seed(seed)

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,)),])

mnist_train = datasets.MNIST(data_path, train=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, transform=transform)

# Create DataLoaders
batch_size = 1
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=0)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=0)

data_in, data_out = 88, 10

def data_pre_process(data):
    data = data.numpy().astype('float64')
    new_data = []
    VS = np.ones((20,), np.float64)
    for VGs in data:
        # rc = tft_rc(VGs, VS, 'value') * 1e5
        Imin = tft(np.zeros((20,), np.float64), VS).ID[-1]
        Imax = tft(np.ones((20,), np.float64), VS).ID[-1]
        # rc = (tft_rc(VGs, VS, 'value', data=VGs, divide=True) - Imin)/(Imax - Imin) + 0.001
        rc = (tft_rc(VGs, VS) - Imin)/(Imax - Imin) + 0.001
        new_data.append(rc)
    new_data = torch.from_numpy(np.array(new_data))
    return new_data


class network(torch.nn.Module):
    def __init__(self):
        super(network, self).__init__()
        self.lin = torch.nn.Linear(data_in, data_out)


    def forward(self, x):
        x = x.reshape(batch_size, 28, 28)
        x_cut = torch.zeros(batch_size, 22,20)
        for _batch in range(batch_size):
            x_cut[_batch] = x[_batch][4:26, 4:24]
        line_in = data_pre_process(x_cut)
        out = self.lin(line_in.float().to(device))
        return out

@timefn
def train_loop(dataloader):
    """
    The model of the neural network will be trained here and update the weights.
    Args:
        dataloader(Dataloader class): Dataset loader from pytorch.

    Returns:
        train_loss
    """
    size = len(dataloader.dataset)
    train_loss = 0.
    train_acc = 0.
    acc_save = []
    for batch, (x,label) in enumerate(dataloader):
        '''
        x: 784
        label: labels
        '''
        out = model(x).to(cpu)
        label_pred = torch.max(out, 1)[1]
        loss = loss_fn(out, label)
        train_loss += loss.item()
        train_correct = (label_pred == label).sum()
        train_acc += train_correct.item()
        #backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % (size // 20) == 0 and batch != 0:
            loss, current = loss.item(), batch * len(x)
            print(f"loss: {loss:>7f}; acc: {train_acc / current *100:.2f}%; [{current:>5d}/{size:>5d}]")
            acc_save.append([train_acc/current, current])
    print(f"accuracy:  [{train_acc / size*100:.2f}%]")
    return train_loss

        
@timefn
def test_loop(dataloader):
    """
    The model of the neural network will be tested here and will not update the weights.

    Args:
        dataloader(Dataloader class): Dataset loader from pytorch.

    Returns:
        test_loss(float)
    """
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    test_loss, correct = 0, 0

    with torch.no_grad():
        for x, label in dataloader:
            out = model(x).to(cpu)
            label_pred = torch.max(out, 1)[1]
            loss = loss_fn(out, label)

            test_loss += loss.item()
            test_correct = (label_pred == label).sum()
            correct += test_correct.item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    return correct

model = network().to(device)
learning_rate = 2e-3   
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)


if __name__ == '__main__':
    os.makedirs(save_path, exist_ok=True)
    train_loss_count = []
    test_accuracy = []
    epoch = num_epochs
    for t in range(epoch + 1):
        print(f'Epoch {t}')
        train_loss_count.append(train_loop(train_loader))
        test_accuracy.append(test_loop(test_loader))
    result = np.array([train_loss_count, test_accuracy])
    torch.save(model.state_dict(), save_path+'/epoch'+str(epoch)+'_net_model_'+str(sigma)+'.pth')
    np.save(save_path+'/epoch'+str(epoch)+'_'+str(sigma)+'.npy', result)
    plt.figure()
    plt.plot(list(range(epoch + 1)), train_loss_count, label='Loss')
    plt.figure()
    plt.plot(list(range(epoch + 1)), test_accuracy, label='Accuracy')
    plt.show()