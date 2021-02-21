import json
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset
from tensorflow.keras import preprocessing
from konlpy.tag import Komoran

class NerDataset(Dataset):
    def __init__(self, type, dict_dir, maxlen):
        super(NerDataset, self).__init__()

        file_name = 'ner_train.txt'

        corpus = self.read_file(file_name)

        sents, tags = self.sepSentTag(corpus)

        self.tag_vocab = preprocessing.text.Tokenizer(lower=False)
        self.tag_vocab.fit_on_texts(tags)
 
        tr_input, val_input, tr_output, val_output = train_test_split(sents,
                                                                      tags,
                                                                      test_size=0.33,
                                                                      random_state=44)
                                                                
        if type == True: # train
            self.sents = tr_input
            self.tags = self.tag_vocab.texts_to_sequences(tr_output)
            
        else: # eval
            self.sents = val_input
            self.tags = self.tag_vocab.texts_to_sequences(val_output)

        if(dict_dir != ''):
            with open(dict_dir, 'rb') as f:
                self.word2idx = json.load(f)

        self.maxlen = maxlen
        self.pad = preprocessing.sequence.pad_sequences

    def __len__(self):
        return len(self.tags)

    def __getitem__(self, idx):
        ids = self.get_ids(self.sents[idx])
        sent = torch.tensor(self.pad([ids], value=0, maxlen=self.maxlen, padding='post'))
        tag = torch.tensor(self.pad([self.tags[idx]], value=0, maxlen=self.maxlen, padding='post'))

        return sent[0], tag[0]
    
    def read_file(self, file_dir):
        sents = []
        with open(file_dir, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            for idx, l in enumerate(lines):
                if l[0] == ';' and lines[idx + 1][0] == '$':
                    this_sent = []
                elif l[0] == '$' and lines[idx - 1][0] == ';':
                    continue
                elif l[0] == '\n':
                    sents.append(this_sent)
                else:
                    this_sent.append(tuple(l.split()))
        return sents

    def sepSentTag(self, corpus):
        sentences, tags = [], []
        for t in corpus:
            sentence = []
            bio_tag = []
            for w in t:
                sentence.append(w[1])
                bio_tag.append(w[3])
            sentences.append(sentence)
            tags.append(bio_tag)
        return sentences, tags
    
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