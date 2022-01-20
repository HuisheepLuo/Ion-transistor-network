import snntorch as snn
from snntorch import spikeplot as splt
from snntorch import spikegen, surrogate
import torch, torch.nn as nn
import snntorch.functional as SF
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.nn.modules import loss
from torch.nn.modules.activation import Softmax
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.animation as animation
import numpy as np


# seed
torch.manual_seed(42)
torch.cuda.manual_seed(42)
np.random.seed(42)

data_path = 'data'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# print(device)
batch_size = 64

num_epochs = 1
num_steps = 200 # run for 25 time steps
step = 0


# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(data_path, train=True, transform=transform)
mnist_test = datasets.MNIST(data_path, train=False, transform=transform)

# Create DataLoaders
train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True, num_workers=2)
test_loader = DataLoader(mnist_test, batch_size=batch_size, shuffle=True, num_workers=2)


# Define Network
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        num_inputs = 784
        num_hidden = 300
        num_outputs = 10
        spike_grad = surrogate.fast_sigmoid()

        # global decay rate for all leaky neurons in layer 1
        beta1 = 0.9
        # independent decay rate for each leaky neuron in layer 2: [0, 1)
        # beta2 = torch.rand((num_outputs), dtype = torch.float) #.to(device)
        beta2 = 0.8

        # Init layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        # self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad, learn_beta=True)
        self.lif1 = snn.Leaky(beta=beta1, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        # self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad,learn_beta=True)
        self.lif2 = snn.Leaky(beta=beta2, spike_grad=spike_grad, output=True)


    def forward(self, x):

        # reset hidden states and outputs at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the final layer
        spk2_rec = []
        mem2_rec = []

        for step in range(num_steps):
            cur1 = self.fc1(x[step].flatten(1))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            spk2_rec.append(spk2)
            mem2_rec.append(mem2)

        return torch.stack(spk2_rec), torch.stack(mem2_rec)

# Load the network onto CUDA if available
net = Net().to(device)

# 
optimizer = torch.optim.Adam(net.parameters(), lr=2e-4, betas=(0.9, 0.999))
# loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)
loss_fn = SF.ce_count_loss()
# loss_fn = loss.CrossEntropyLoss()


def train_loop(train_loader, net, num_steps):
    for i, (data, targets) in enumerate(iter(train_loader)):
        data = data.to(device)
        targets = targets.to(device)
        spike_data = spikegen.rate(data, num_steps=num_steps)
        net.train()
        spk_rec, mem_rec = net(spike_data)
        spk_count = torch.sum(spk_rec)
        loss_val = loss_fn(spk_rec, targets)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # print every 25 iterations
        if i % 25 == 0:
            net.eval()
            print(f"Epoch {epoch}, Iteration {i} \nTrain Loss: {loss_val.item():.2f}")
            print(f"spk_count:{spk_count}")
            # check accuracy on a single batch
            acc = SF.accuracy_rate(spk_rec, targets)
            acc_hist.append(acc)
            print(f"Accuracy: {acc * 100:.2f}%\n")

        # uncomment for faster termination
        # if i == 150:
        #     break

    
def test_accuracy(data_loader, net, num_steps):
    with torch.no_grad():
        total = 0
        acc = 0
        net.eval()
        data_loader = iter(data_loader)
        for data, targets in data_loader:
            data = data.to(device)
            targets = targets.to(device)
            spike_data = spikegen.rate(data, num_steps=num_steps)
            spk_rec, _ = net(spike_data)

            acc += SF.accuracy_rate(spk_rec, targets) * spk_rec.size(1)
            total += spk_rec.size(1)

    return acc/total

def update(spike_data, *args):
    global num_steps, step
    if step < num_steps:
        im = plt.imshow(spike_data[step][0].reshape((28,28)), cmap='gray', animated=True)
        step += 1
    else: 
        step = 0
        im = plt.imshow(spike_data[step][0].reshape((28,28)), cmap='gray', animated=True)

    return im,

def display_ani_spk():
    for data, targets in iter(train_loader):
        spike_data = spikegen.rate(data, num_steps=num_steps)
        # spike_data = spikegen.latency(data, num_steps=num_steps, tau=5, threshold=0.01)

        # animation generate
        fig = plt.figure()
        im = plt.imshow(spike_data[0][0].reshape((28,28)), cmap='gray', animated=True)
        ann = animation.FuncAnimation(fig, update, interval=60, blit = True, cache_frame_data=False)
        ann.save('mnist_spike_latency_code.gif')
        plt.show()
    return spike_data

if __name__ == '__main__':
    loss_hist = []
    acc_hist = []

    # training loop
    for epoch in range(num_epochs):
        train_loop(train_loader, net, num_steps)
        test_acc = test_accuracy(test_loader, net, num_steps)

        print(f"Test Accuracy: {test_acc * 100:.2f}%\n")

    torch.save(net.state_dict(), 'spk_net_model_batch_64_hidden_300.pth')
    plt.rcParams['font.size'] = 14
    plt.rcParams['font.sans-serif'] = ['Arial']
    result = np.array([loss_hist, acc_hist])
    np.save('result_spk_steps_200_seed_42_batch_64_hidden_300.npy', result)
    fig = plt.figure(facecolor="w", figsize=(8, 4))
    plt.plot(loss_hist)
    plt.title("Loss Curves")
    plt.legend("Train Loss")
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()



