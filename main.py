import os
import time
import torch
import pickle
import argparse

import numpy as np

from modules import ListenAttendSpell
from dataset.dataloader import load_data, get_loader
from torch.autograd import Variable
from metric import seq_to_sen, WER

def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device :", device)
    # Load Dataset
    speech_data = load_data(args.path, args.test, False)

    with open(os.path.join('libri_fbank40_char30', 'mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)

    ###
    seed = 0
    lr = 0.2
    decay = 0.98
    start_epoch = 0
    beam_size = 16
    bst_loss = 1e9

    eos_idx = 1
    pad_idx = 2

    filter_bank_size = 40
    hid_dim = 512
    char_dim = len(mapping.keys())

    beam_search = True

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Model initialization
    model = ListenAttendSpell(filter_bank_size, hid_dim, args.batch_size, char_dim, device).to(device)

    # SET OPTIMIZER
    optimizer = torch.optim.ASGD(model.parameters(), lr=lr)

    criterion = torch.nn.NLLLoss(ignore_index=pad_idx).to(device)

    # Load checkpoint
    save_path = "checkpoint/best.tar"

    if args.model_load:
        assert os.path.exists(save_path)
        ckpt = torch.load(save_path)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])
        start_epoch = ckpt["epoch"]
        print("Model Loaded : Trained for %d epochs." %(start_epoch))

    if not args.test:
        # split dataset into train and valid
        total_data_len = len(speech_data)
        test_idx = int(total_data_len*10/2000)
        train_data, valid_data = speech_data[test_idx:], speech_data[:test_idx]

        train_loader = get_loader(args.path, train_data, args.batch_size, shuffle=True)
        valid_loader = get_loader(args.path, valid_data, args.batch_size, shuffle=True)

        itr = 0

        for epoch in range(start_epoch, args.epochs):
            epoch_start = time.time()
            # TRAIN STEP
            for src_batch, tgt_batch in train_loader:
                src_batch = src_batch.astype(np.double)
                tgt_batch = tgt_batch.astype(np.double)
                src_batch = Variable(torch.FloatTensor(src_batch)).to(device)
                tgt_batch = Variable(torch.LongTensor(tgt_batch)).to(device)

                model.train()
                optimizer.zero_grad()

                output = model(src_batch, tgt_batch)
                m, l, h = output.size()
                output = output.contiguous().view(m*l, h)
                gt = tgt_batch.view(-1)

                loss = criterion(output, gt)
                loss.backward()
                optimizer.step()
                if (itr % 100) == 0:
                    print("TRAIN LOSS : %f" %(loss.item()))
                itr += 1

            # VALID STEP
            for src_batch, tgt_batch in valid_loader:
                model.eval()
                src_batch = src_batch.astype(np.double)
                tgt_batch = tgt_batch.astype(np.double)
                src_batch = Variable(torch.FloatTensor(src_batch)).to(device)
                tgt_batch = Variable(torch.LongTensor(tgt_batch)).to(device)

                with torch.no_grad():
                    output = model(src_batch, tgt_batch)
                    m, l, h = output.size()
                    output = output.contiguous().view(m*l, h)
                    gt = tgt_batch.view(-1)

                    loss = criterion(output, gt)
                print("EPOCH %d. Validation Loss %f. Time takes %s" %(epoch, loss.item(), time.time()-epoch_start))

            # SAVE MODEL CHECKPOINT
            if loss < bst_loss :
                if (not os.path.exists("checkpoint")):
                    os.mkdir("checkpoint")
                torch.save({"epoch": epoch, "model":model.state_dict(), "optim":optimizer.state_dict()}, save_path)
                bst_loss = loss
    else:
        # TEST STEP
        test_loader = get_loader(args.path, speech_data, args.batch_size, shuffle=True)
        gt = []
        generated = []
        start = time.time()

        max_len = 1000

        for src_batch, tgt_batch in test_loader:
            model.eval()
            src_batch = src_batch.astype(np.double)
            tgt_batch = tgt_batch.astype(np.double) # [batch_size][seq_len]
            src_batch = Variable(torch.FloatTensor(src_batch)).to(device)
            tgt_batch = Variable(torch.LongTensor(tgt_batch)).to(device) 
            
            y = torch.zeros(1, 1).long().to(device)

            with torch.no_grad():   
                if (beam_search):
                    # BEAM SEARCH INFERENCE
                    sentences = model.predict(src_batch, y, beam_size, eos_idx)
                    tgt_batch = tgt_batch[0].unsqueeze(dim=0).tolist()
                else:
                    # GREEDY SEARCH INFERENCE
                    sentences = model.greedy_predict(src_batch, y, max_len, eos_idx)
                    tgt_batch = tgt_batch.tolist()
                sentences = sentences.tolist()
                generated = generated + sentences
                gt = gt + tgt_batch
            # print("finish one iter")

        wer = WER(generated, gt)
        print("BEAM SEARCH : %d" %(beam_search))
        print("WER %f" %(wer))
        print("TIME CONSUMED %s" %(time.time() - start))

        sentences = seq_to_sen(generated)
        write_sen = []
        for line in sentences:
            tmp = ''.join(line)
            write_sen.append(tmp)
        with open("asr_texts.txt", 'w+') as f:
            for line in write_sen:
                f.write(line+"\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Transformer')
    parser.add_argument(
        '--path',
        type=str,
        default='processed_data')

    parser.add_argument(
        '--epochs',
        type=int,
        default=50)
    parser.add_argument(
        '--batch_size',
        type=int,
        default=32)

    parser.add_argument(
        '--test',
        action='store_true')

    parser.add_argument(
        '--model_load',
        action='store_true')
    args = parser.parse_args()

    main(args)