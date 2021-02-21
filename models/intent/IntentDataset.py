import json

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from tensorflow.keras import preprocessing
from konlpy.tag import Komoran

class IntentDataset(Dataset):
    def __init__(self, type, dict_dir, userDict_dir, maxlen):
        super(IntentDataset, self).__init__()

        train_file = "total_train_data.csv"

        data = pd.read_csv(train_file, delimiter=',')

        queries, intents = list(data['query']), list(data['intent'])

        tr_input, val_input, tr_output, val_output = train_test_split(queries, 
                                                                      intents, 
                                                                      train_size=0.33, 
                                                                      random_state=44)

        if type == True: # train
            self.queries = tr_input
            self.intents = tr_output

        else: # eval
            self.queries = val_input
            self.intents = val_output

        if(dict_dir != ''):
            with open(dict_dir, 'rb') as f:
                self.word2idx = json.load(f)
       
        komoran = Komoran(userdic=userDict_dir)
        self.queries_pos = []
        for i in range(len(self.queries)):
            self.queries_pos.append(komoran.pos(self.queries[i]))

        self.maxlen = maxlen
        self.pad = preprocessing.sequence.pad_sequences
        # 제외할 품사(관계언, 기호, 어미, 접미사)
        self.exclusion_tags = [
            'JKS', 'JKC', 'JKG', 'JKO', 'JKB', 'JKV', 'JKQ',
            'JX', 'JC',
            'SF', 'SP', 'SS', 'SE', 'SO',
            'EP', 'EF', 'EC', 'ETN', 'ETM',
            'XSN', 'XSV', 'XSA'
        ]

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        keywords = self.get_keywords(self.queries_pos[idx])
        ids = self.get_ids(keywords)
        query = torch.tensor(self.pad([ids], value=0, padding='post', maxlen=self.maxlen))

        intent = torch.tensor(self.intents[idx])

        return query[0], intent
    
    def get_keywords(self, pos, without_tag=True):
        f = lambda x: x in self.exclusion_tags
        keywords = []
        for p in pos:
            if f(p[1]) is False:
                keywords.append(p if without_tag is False else p[0])
        return keywords

    def get_ids(self, words):
        if self.word2idx is None:
            return []
        ids = []
        for word in words:
            try:
                ids.append(self.word2idx[word])
            except:
                ids.append(self.word2idx['OOV'])
        return ids




        




    

    

