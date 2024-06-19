import torch.nn as nn 
import torch
from embed.embed import DataEmbedding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu") 

class ATTLSTM(nn.Module):

    def __init__(self):
        super(ATTLSTM, self).__init__()
        self.embed = DataEmbedding(c_in=3, d_model=32)
        self.transencoderlayer = nn.TransformerEncoderLayer(d_model=32, nhead=4,
                                                             dim_feedforward=64, batch_first=True)
        self.encoder = nn.TransformerEncoder(self.transencoderlayer, num_layers=1)
        self.decoder = nn.LSTM(input_size= 32, hidden_size= 64, batch_first= True)
        self.dense1 = nn.Sequential(
            nn.Linear(in_features=64, out_features= 1),
            nn.Dropout(p = 0.01)
        )
    def forward(self, x):
        output1 = self.embed(x)
        output2 = self.encoder(output1)
        output3, _ = self.decoder(output2)
        fin_output = self.dense1(output3)

        return fin_output
    
if __name__ =='__main__':
    print(ATTLSTM())