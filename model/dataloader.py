import torch 
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.preprocessing import StandardScaler 
from model.timefeature import timefeature


class windpowerdataset(Dataset):
    def __init__(self, data_object,  pred_len= 48) :
        self.len = data_object.shape[0]
        self.pdata = data_object[:,:,0]
        self.pdata_stamp = data_object[:,:,2:]
        self.pred_len = pred_len
        pass

    def __getitem__(self, index):
        x_seq = self.pdata[index, :-self.pred_len]
        y_seq = self.pdata[index, -(self.pred_len):]
        x_seq_mark = self.pdata_stamp[index,  :-self.pred_len, :]
        y_seq_mark = self.pdata_stamp[index, -(self.pred_len):, :]

        x_seq = x_seq.reshape(-1,3)      
        y_seq = np.expand_dims(y_seq, axis = 1)

        x_seq_mark = timefeature(x_seq_mark)
        y_seq_mark = timefeature(y_seq_mark)

        return x_seq, y_seq, x_seq_mark, y_seq_mark
    
    def __len__(self):
        return self.len

if __name__ == '__main__':
    folderpath = './4types_data/dataset3.npy'
    data0 = np.load(folderpath)
    dataset = windpowerdataset(data0, 96, 0)
    trainloader = DataLoader(dataset, 2, False, num_workers=0)
    for i, (value1, value2, value3, value4) in enumerate(trainloader):
        continue