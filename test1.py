import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot  as plt
device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

def mean_absolute_error(y_true, y_pred):
    return torch.mean(torch.abs(y_true - y_pred))

def test(model,  trainloader):

    model.eval()
    with torch.no_grad():
        loss_fn = nn.MSELoss()
        
        mae_total_loss = []
        rmse_total_loss = []
        for i,(x,y,_,_) in enumerate(trainloader):
            x = x.float().to(device)
            y = y.float().to(device)

            output = model(x)

            loss = loss_fn(output, y)
            sqrt_loss = torch.sqrt(loss)
            rmse_total_loss.append(sqrt_loss.item())
            mae_total_loss.append(mean_absolute_error(y, output).item())
            # if i%5 == 0:
            #     a = output[-1].cpu().squeeze().numpy()
            #     b = y[-1].cpu().squeeze().numpy()
            #     plt.plot(a, label='pred')
            #     plt.plot(b, label ='true')
            #     plt.legend()
            #     plt.show()
        return np.average(rmse_total_loss), np.average(mae_total_loss)