import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def train(model,  trainloader):

    model.train()
    loss_fn = nn.MSELoss()
    opt = torch.optim.Adam(model.parameters(), lr= 0.001)
    
    total_loss = []
    for i,(x,y,_,_) in enumerate(trainloader):
        x = x.float().to(device)
        y = y.float().to(device)

        output = model(x)

        loss = loss_fn(output, y)
        total_loss.append(loss.item())

        opt.zero_grad()
        loss.backward()
        opt.step()

    return np.average(total_loss)