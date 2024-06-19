import numpy as np 
from sklearn.preprocessing import MinMaxScaler

class DatasetCreator:
    def __init__(self, dataset, labels, window_length=3, points_per_day=96):
        self.scaler =   MinMaxScaler()
        self.data = dataset
        self.labels = labels        
        self.window_length = window_length
        self.points_per_day = points_per_day

    def create_subsets(self):        
        subsets = [[] for _ in range(max(self.labels) + 1)]  # Create a list of empty lists for each label type
        # self.data = self.scaler.fit_transform(self.data)
        for index, label in enumerate(self.labels):
            if index < self.window_length:
                continue
            past3day = self.data[(index-3)*96+24:(index-3)*96+72,:]
            past2day = self.data[(index-2)*96+24:(index-2)*96+72,:]
            past1day = self.data[(index-1)*96+24:(index-1)*96+72,:]
            today = self.data[index*96+24:index*96+72,:]
            subset_data = np.concatenate([past3day,past2day,past1day,today])
            subsets[label].append(subset_data)
        return subsets
