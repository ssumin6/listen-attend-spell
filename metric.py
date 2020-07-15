import os
import pickle

def seq_to_sen(seq):
    with open(os.path.join('libri_fbank40_char30', 'mapping.pkl'), 'rb') as f:
        mapping = pickle.load(f)

    reverse_map = {}
    for key in mapping.keys():
        value = mapping[key]
        reverse_map[value] = key
    
    # seq shape : [batch_size, seq]
    # Assume that seq input is python list
    sentences = []
    for line in seq:
        if ('<eos>' not in line):
            eos_idx = -1
        else:
            eos_idx = line.index('<eos>')
        if ('<sos>' not in line):
            sos_idx = 0
        else:
            sos_idx = line.index('<sos>')
        tmp = [reverse_map[int(idx)] for idx in line[sos_idx+1:eos_idx]]
        sentences.append(tmp)

    return sentences

def same(src_sen, gt_sen):
    if (len(src_sen)!=len(gt_sen)):
        return False
    for i in range(len(src_sen)):
        if (src_sen[i] != gt_sen[i]):
            return False
    return True

def WER(src, gt):
    '''
    wer : word errors (S+D+I)
    N : total words
    
    src : 2d array
    gt : 2d array
    '''
    # return percentage
    N, wer = 0, 0

    src = seq_to_sen(src)
    gt = seq_to_sen(gt)

    srcs = []
    for line in src: 
        tmp = ''.join(line).split(' ')
        srcs.append(tmp)
    src = srcs
    # print(src)
    gts = []
    for line in gt: 
        tmp = ''.join(line).split(' ')
        gts.append(tmp)
    gt = gts
    # print(gt)

    for idx in range(len(gt)):
        N += len(gt[idx])

        gt_sen = gt[idx]
        src_sen = src[idx]

        if (same(src_sen, gt_sen)):
            continue 
        
        d = [[0 for _ in range(len(gt_sen)+1)] for _ in range(len(src_sen)+1)]
        for i in range(len(gt_sen)+1):
            d[0][i] = i
        for i in range(len(src_sen)+1):
            d[i][0] = i
        
        for i in range(1, len(src_sen)+1):
            for j in range(1, len(gt_sen)+1):
                if src_sen[i-1]==gt_sen[j-1]:
                    d[i][j] = d[i-1][j-1]
                else:
                    S = d[i-1][j-1]+1
                    I = d[i][j-1]+1
                    D = d[i-1][j] +1
                    d[i][j] = min(S, I, D)
        wer += d[len(src_sen)][len(gt_sen)]

    return 1.*(wer)/N