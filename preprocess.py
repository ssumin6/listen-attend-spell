import os
import argparse

import numpy as np
from dataset.dataloader import load_data, get_loader, shuffle_list

def load_and_pad(src_batch, max_len, path):
    data = []
    for filename in src_batch:
        sample = np.load(os.path.join(path, filename))
        data.append(sample) #(length, 40)

    for i in range(len(data)):
        data[i] = np.pad(data[i], ((0, max_len),(0,0)), 'constant', constant_values=0)[:max_len][:]
    return np.asarray(data)
    
def pad(batch, max_len):
    for i in range(len(batch)):
        batch[i] += [2] * (max_len - len(batch[i]))
    return np.asarray(batch)

def batch_processing(batch, save_path, idx, file_read_path):
    lengths = []
    src_batch = []
    tgt_batch = []
    
    for path, labels, length in batch:
        src_batch.append(path)
        tmp = labels.split('_')
        tgt_batch.append(tmp)
        lengths.append(length)

    max_length = ((max(lengths) // 8)+1)*8

    src_batch = load_and_pad(src_batch, max_length, file_read_path)
    tgt_batch = pad(tgt_batch, max_length)

    src_file = os.path.join(save_path, "speech%03d.npy" %idx)
    np.save(src_file, src_batch)
    tgt_file = os.path.join(save_path, "labels%03d.npy" %idx)
    np.save(tgt_file, tgt_batch)

    return src_file, tgt_file

def main(args):
    dataset = load_data(args.path, args.test)
    new_file_set = []
    size = len(dataset)
    print("DATASET SIZE: %d" % size)

    path = "processed_data"
    if (not os.path.exists(path)):
        os.mkdir(path)

    if (not args.test):
        csv_file = 'train-clean-100.csv'
    else:
        csv_file = 'test-clean.csv'
    
    subpath = "train" if not args.test else "test"
    subpath = os.path.join(path, subpath)
    if (not os.path.exists(subpath)):
        os.mkdir(subpath)

    index = 0
    while (index>=0):
        if args.batch_size * index >= size:
            break
        
        batch = dataset[args.batch_size*index : args.batch_size*(index+1)]
        batch_len = len(batch)

        src_file, tgt_file = batch_processing(batch, subpath, index, args.path)
        length = min(args.batch_size, batch_len)

        new_file_set.append([src_file, tgt_file, str(length)])

        index += 1
    with open(os.path.join(path, csv_file), "w+") as f:
        for line in new_file_set:
            f.write(','.join(line)+'\n')
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='libri_fbank40_char30')

    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)

    parser.add_argument(
        '--test',
        action='store_true')

    args = parser.parse_args()

    main(args)