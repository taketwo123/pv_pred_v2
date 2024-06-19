import torch
from model.ATTLSTM import ATTLSTM
import numpy as np
from model.dataloader import windpowerdataset
from torch.utils.data import DataLoader
import matplotlib
matplotlib.rc('font', family = 'simsun')
import matplotlib.pyplot as plt
from train import train
from test1 import test

data = np.load('./data/dataset2.npy')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

boader1 = round(data.shape[0]*0.8)
traindata = data[:boader1]
testdata = data[boader1:]

trainset = windpowerdataset(traindata)
testset = windpowerdataset(testdata)
trainloader = DataLoader(trainset, batch_size=30, shuffle= True)
testloader = DataLoader(testset, batch_size=20, shuffle= True)

model = ATTLSTM().to(device)

train_total_loss = []
for epoch in range(100):

    train_loss = train(model, trainloader)
    train_total_loss.append(train_loss)
    print("-"*25, f"Epoch {epoch + 1}","-"*25)
    print(f"Training loss: {train_loss:.4f}")

plt.plot(train_total_loss)
plt.xlabel('training epochs')
plt.ylabel('MAEloss')
plt.title('model_2')
plt.show()

# rmse_loss, mae_loss = test(model, testloader)
# print(f"rmse_loss: {rmse_loss:.4f} mae_loss: {mae_loss:.4f}")

# torch.save(model.state_dict(),'./result/model2.pt')

