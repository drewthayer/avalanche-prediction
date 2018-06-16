import pickle
import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import matplotlib.pyplot as plt


class  Net(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(50, 50)
        self.relu1 = nn.ReLU()
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(50, 100)
        self.prelu = nn.PReLU(1)
        self.out = nn.Linear(100, 1)
        self.out_act = nn.Sigmoid()

    def forward(self, input_):
        a1 = self.fc1(input_)
        h1 = self.relu1(a1)
        dout = self.dout(h1)
        a2 = self.fc2(dout)
        h2 = self.prelu(a2)
        a3 = self.out(h2)
        y = self.out_act(a3)
        return y

def train_epoch(X, y, model, opt, criterion, batch_size=50):
    model.train()
    losses = []
    for beg_i in range(0, X.size(0), batch_size):
        x_batch = X[beg_i:beg_i + batch_size, :]
        y_batch = y[beg_i:beg_i + batch_size]
        x_batch = Variable(x_batch, volatile=True)
        y_batch = Variable(y_batch)

        opt.zero_grad()
        # (1) Forward
        y_hat = net(x_batch)
        # (2) Compute diff
        loss = criterion(y_hat, y_batch)
        # (3) Compute gradients
        loss.backward()
        # (4) update weights
        opt.step()
        losses.append(loss.data.numpy())
    return losses


if __name__=='__main__':
    zonename = sys.argv[1] # 'aspen' or 'nsj'
    case_select = sys.argv[2] # 'slab' or 'wet'

    # read engineered X,y data from pickle
    Xarray, yvect = pickle.load(
    open('feature-matrices/{}_{}_matrices.pkl'.format(zonename, case_select),
    'rb'))

    # implement in pytorch
    X = torch.FloatTensor(Xarray)
    y = torch.tensor(yvect)

    net = Net()
    opt = optim.Adam(net.parameters(), lr=0.001, betas=(0.9, 0.999))
    criterion = nn.BCELoss()

    e_losses = []
    num_epochs = 20
    for e in range(num_epochs):
        e_losses += train_epoch(X, y, net, opt, criterion)
    plt.plot(e_losses)
