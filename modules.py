import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

def is_finished(seq, eos_token):
    seq = seq.tolist()
    for line in seq:
        if eos_token not in line:
            return False
    return True

def compute_mask(seq, seq_len):
    ##### compute mask for masked_fill
    bs, ts, hs = seq.shape
    mask = torch.zeros(0)
    mask = np.zeros((bs, ts, hs))
    for idx, sl in enumerate(seq_len):
        mask[sl:, :] = 1  
    mask = torch.from_numpy(mask).to(
        seq_len.device, dtype=torch.bool)  # BNxT
    return mask

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

    def forward(self, s, h, mask= None):
        s = s.unsqueeze(dim=-1)
        if mask is not None:
            h = h.masked_fill(mask, -1e9)
        alpha = torch.bmm(h, s)
        alpha = F.softmax(alpha, dim=1)
        alpha = alpha.transpose(1, 2)
        c = torch.bmm(alpha, h).squeeze(dim=1)
        return c

class Listener(nn.Module):
    def __init__(self, filterbanksize, hidden_dim, batch_size):
        super(Listener, self).__init__()
        self.batch_size = batch_size
        self.filterbanksize = filterbanksize
        self.hidden_dim = hidden_dim
        self.pblstm_layer = 3
        # 3 pBLSTM layers over 1 BLSTM
        self.blstm = nn.LSTM(self.filterbanksize, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        self.pblstm = nn.ModuleList([nn.LSTM(self.hidden_dim*4, self.hidden_dim, num_layers=1, batch_first=True, bidirectional=True) for _ in range(self.pblstm_layer)])

    def forward(self, x, lgt_batch):
        x, _ = self.blstm(x)
        for i in range(self.pblstm_layer):
            batch_size, seq_len, input_size = x.size()
            x = x.contiguous().view(batch_size, seq_len//2, input_size*2)
            lgt_batch = lgt_batch // 2
            x, _ = self.pblstm[i](x)
        return x, lgt_batch

class Speller(nn.Module):
    def __init__(self, batch_size, hid_dim, out_dim, device):
        super(Speller, self).__init__()
        self.batch_size = batch_size
        self.out_dim = out_dim
        self.hid_dim = hid_dim
        self.device = device
        self.embed = nn.Embedding(self.out_dim, self.hid_dim)
        self.attention = Attention()
        self.lstm1 = nn.LSTMCell(self.hid_dim*2, self.hid_dim)
        self.lstm2 = nn.LSTMCell(self.hid_dim, self.hid_dim)
        self.mlp = nn.Linear(self.hid_dim*2, self.out_dim)

    def forward(self, hidden, y, enc_len):
        """
        y : shape [batch_size, seq_len]
        """
        y = self.embed(y)
        seq_len = y.size()[1]
        y = y.transpose(0, 1) # change the batch and seq_lens

        batch_size = hidden.size()[0]

        outputs = []
        s = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        c1 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        c2 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        s2 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)

        # weight initialization with uniform function
        nn.init.uniform_(s, a=-0.1, b=0.1)
        nn.init.uniform_(c1, a=-0.1, b=0.1)
        nn.init.uniform_(c2, a=-0.1, b=0.1)
        nn.init.uniform_(s2, a=-0.1, b=0.1)

        mask = compute_mask(hidden, enc_len)

        c = self.attention(s, hidden, mask=mask) # shape : [batch_size][hid_dim]

        for i in range(seq_len):
            # 1st layer of LSTM
            s, c1 = self.lstm1(torch.cat((y[i], c), dim=1), (s, c1))
            # 2nd layer of LSTM
            s2, c2 = self.lstm2(s, (s2, c2))
            c = self.attention(s2, hidden, mask=mask) # shape : [batch_size][hid_dim]
            output = F.log_softmax(self.mlp(torch.cat((s2, c), dim=1)), dim=-1)
            outputs.append(output)
        outputs = torch.stack(outputs, dim=0).transpose(0, 1)
        return outputs

    def predict(self, enc_outputs, y, enc_len, beam_size, eos_token):
        # FOR BEAM SEARCH INFERENCE

        enc_outputs = enc_outputs.unsqueeze(dim=0)
        # enc_outputs [batch_size][different seq_len][hid_dim]
        maxlen = enc_outputs.size()[1] 

        batch_size = 1

        s = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        c1 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        c2 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)
        s2 = torch.empty(batch_size, self.hid_dim, requires_grad=True).to(self.device)

        # weight initialization with uniform function
        nn.init.uniform_(s, a=-0.1, b=0.1)
        nn.init.uniform_(c1, a=-0.1, b=0.1)
        nn.init.uniform_(c2, a=-0.1, b=0.1)
        nn.init.uniform_(s2, a=-0.1, b=0.1)

        mask = compute_mask(enc_outputs, enc_len)

        c = self.attention(s, enc_outputs, mask=mask) # shape : [batch_size][hid_dim]
        vy = torch.zeros(1, 1).long().to(self.device)

        hyp = {'score': 0.0, 'yseq': [y.item()], 'c_prev': [c1, c2], 'h_prev': [s, s2],
               'a_prev': c}
        hyps = [hyp]
        ended_hyps = []

        for i in range(maxlen):
            hyps_best_kept = []
            for hyp in hyps:
                vy[0][0] = hyp['yseq'][i]
                c = hyp['a_prev']
                s, s2 = hyp['h_prev']
                c1, c2 = hyp['c_prev']
                embedded = self.embed(vy)
                embedded = embedded.transpose(0, 1)
                s, c1 = self.lstm1(torch.cat((embedded[0], c), dim=1), (s, c1))
                s2, c2 = self.lstm2(s, (s2, c2))
                c = self.attention(s2, enc_outputs, mask=mask) # shape : [1][hid_dim]
                local_scores = F.log_softmax(self.mlp(torch.cat((s2, c), dim=1)), dim=-1)
                # topk scores
                local_best_scores, local_best_ids = torch.topk(
                    local_scores, beam_size, dim=1)

                for j in range(beam_size):
                    new_hyp = {}
                    new_hyp['h_prev'] = [s, s2]
                    new_hyp['c_prev'] = [c1, c2]
                    new_hyp['a_prev'] = c[:]
                    new_hyp['score'] = hyp['score'] + local_best_scores[0, j]
                    new_hyp['yseq'] = [0] * (1 + len(hyp['yseq']))
                    new_hyp['yseq'][:len(hyp['yseq'])] = hyp['yseq']
                    new_hyp['yseq'][len(hyp['yseq'])] = int(
                        local_best_ids[0, j])
                    # will be (2 x beam) hyps at most
                    hyps_best_kept.append(new_hyp)

                hyps_best_kept = sorted(hyps_best_kept,
                                        key=lambda x: x['score'],
                                        reverse=True)[:beam_size]
            # end for hyp in hyps
            hyps = hyps_best_kept

            # add eos in the final loop to avoid that there are no ended hyps
            if i == maxlen - 1:
                for hyp in hyps:
                    hyp['yseq'].append(eos_token)

            # add ended hypothes to a final list, and removed them from current hypothes
            # (this will be a probmlem, number of hyps < beam)
            remained_hyps = []
            for hyp in hyps:
                if hyp['yseq'][-1] == eos_token:
                    # hyp['score'] += (i + 1) * penalty
                    ended_hyps.append(hyp)
                else:
                    remained_hyps.append(hyp)

            hyps = remained_hyps
            if (len(hyps) <= 0):
                break
        # end for i in range(maxlen)

        nbest_hyps = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[:1]
        return nbest_hyps


class ListenAttendSpell(nn.Module):
    def __init__(self, filterbanksize, hid_dim, batch_size, char_dim, device):
        super(ListenAttendSpell, self).__init__()
        self.filterbanksize = filterbanksize
        self.hid_dim = hid_dim
        self.device = device
        self.batch_size = batch_size
        self.dec_out_dim = char_dim
        # Encoder has dimension 256
        self.listener = Listener(self.filterbanksize, self.hid_dim // 2, self.batch_size)
        self.speller = Speller(self.batch_size, self.hid_dim, self.dec_out_dim, self.device)

    def forward(self, x, y, lgt_batch):
        x, enc_len = self.listener(x, lgt_batch)
        x = self.speller(x, y, enc_len)
        return x

    def greedy_predict(self, x, y, lgt_batch, max_len, eos_token):
        # FOR GREEDY INFERENCE
        enc_outputs, enc_len = self.listener(x, lgt_batch)
        pred_batches = self.speller(enc_outputs, y, enc_len)
        sentences = torch.argmax(pred_batches, dim =-1)
        while (pred_batches.size()[1] < max_len and not is_finished(sentences, eos_token)):
            new_input = torch.cat((y, sentences), dim=1)
            pred_batches = self.speller(enc_outputs, new_input)
            sentences = torch.argmax(pred_batches, dim=-1)
        
        return sentences

    def predict(self, x, y, lgt_batch, beam_size, eos_token):
        # FOR BEAM SEARCH INFERENCE
        enc_outputs, enc_len = self.listener(x, lgt_batch)
        hypos = self.speller.predict(enc_outputs[0], y, enc_len, beam_size, eos_token)
        hypos = torch.Tensor(list(map(lambda x:x['yseq'], list(hypos))))
        return hypos