from torch.utils.data import Dataset
import pickle as pkl
import torch
import pandas as pd
from pandas import DataFrame as df


class MovieRankDataset(Dataset):
    
    def __init__(self, pkl_file, args, drop_dup=False):
        
        df = pkl.load(open(pkl_file, 'rb'))
        self.args = args
        if drop_dup == True:
            df_user = df.drop_duplicates(['user_id'])
            df_movie = df.drop_duplicates(['movie_id'])
            self.dataFrame = pd.concat((df_user, df_movie), axis=0)
        else:
            self.dataFrame = df
    
    def __len__(self):
        return len(self.dataFrame)
    
    def __getitem__(self, idx):
        
        # user data
        uid = self.dataFrame.iloc[idx, 0]
        gender = self.dataFrame.iloc[idx, 3]
        age = self.dataFrame.iloc[idx, 4]
        job = self.dataFrame.iloc[idx, 5]
        
        # movie data
        mid = self.dataFrame.iloc[idx, 1]
        mtype = self.dataFrame.iloc[idx, 7]
        mtext = self.dataFrame.iloc[idx, 6]
        
        # target
        rank = torch.FloatTensor([self.dataFrame.iloc[idx, 2]]).to(self.args.device)
        user_inputs = {
            'uid': torch.LongTensor([uid]).view(1, -1).to(self.args.device),
            'gender': torch.LongTensor([gender]).view(1, -1).to(self.args.device),
            'age': torch.LongTensor([age]).view(1, -1).to(self.args.device),
            'job': torch.LongTensor([job]).view(1, -1).to(self.args.device)
        }
        
        movie_inputs = {
            'mid': torch.LongTensor([mid]).view(1, -1).to(self.args.device),
            'mtype': torch.LongTensor(mtype).to(self.args.device),
            'mtext': torch.LongTensor(mtext).to(self.args.device)
        }
        
        sample = {
            'user_inputs': user_inputs,
            'movie_inputs': movie_inputs,
            'target': rank
        }
        return sample
