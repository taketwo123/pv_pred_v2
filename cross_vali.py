import torch
from model.ATTLSTM import ATTLSTM
import numpy as np
from model.dataloader import windpowerdataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.rc('font', family = 'simsun')
import matplotlib.pyplot as plt
from test1 import test

data = np.load('./data/dataset1.npy')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

boader1 = round(data.shape[0]*0.8)
testdata = data[boader1:]

testset = windpowerdataset(testdata)
testloader = DataLoader(testset, batch_size=20, shuffle= True)

model = ATTLSTM().to(device)
model.load_state_dict(torch.load("./result/model2.pt"))

with torch.no_grad():
    rmse_loss, mae_loss = test(model, testloader)
    print(f"rmse_loss: {rmse_loss:.4f} mae_loss: {mae_loss:.4f}")