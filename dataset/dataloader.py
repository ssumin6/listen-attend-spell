import os
import random
import numpy as np
import matplotlib.pyplot as plt

#### PREPROCESS를 따로 할 수 있으면 좋을 듯 데이터셋이 커서

def load_data(path, test=False, preprocess=True):
    data = []
    
    if not test:
        file_name = 'train-clean-100.csv'
    else:
        file_name = 'test-clean.csv'
        # file_name = "train-test-set.csv"
    

    with open (os.path.join(path, file_name)) as f:
        csv = f.readlines()
        for line in csv[1:]:
            line = line.strip().split(',')
            if (preprocess):
                data.append([line[1], line[2], int(line[3])]) # [file_path, labels, length]
            else:
                data.append([line[0], line[1], line[2]]) # [file_path, labels, length_file]

    return data

def shuffle_list(dataset):
    index = list(range(len(dataset)))
    random.shuffle(index)

    shuffle = []

    for i in index:
        shuffle.append(dataset[i])

    return shuffle

class DataLoader:
    def __init__(self, path, dataset, batch_size, pad_idx, shuffle=False):
        self.dataset = dataset
        self.size = len(dataset)
        self.path = path
        self.batch_size = batch_size
        self.pad_idx = pad_idx
        self.shuffle = shuffle

    def __iter__(self):
        self.index = 0

        if self.shuffle:
            self.dataset = shuffle_list(self.dataset)

        return self

    def load(self, batch):
        src_path, tgt_path, lgt_path = batch[0]
        src_batch = np.load(src_path)
        tgt_batch = np.load(tgt_path)
        lgt_batch = np.load(lgt_path)

        return src_batch, tgt_batch, lgt_batch

    def __next__(self):
        if self.index >= self.size:
            raise StopIteration
        
        batch = self.dataset[self.index : self.index+1]

        src_batch, tgt_batch, lgt_batch = self.load(batch)

        self.index += 1

        return src_batch, tgt_batch, lgt_batch

def get_loader(path, dataset, batch_size, shuffle=False):
    data_loader = DataLoader(path, dataset, batch_size=batch_size, pad_idx=2, shuffle=shuffle)

    return data_loader
