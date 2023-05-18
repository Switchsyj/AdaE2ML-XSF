import argparse
import logging

import torch
import torch.nn as nn
import numpy as np
import os
import random

class BiRNNLM(nn.Module):
  def __init__(self, vocab_size, bidirectional=True, dropout=0.5):
    super(BiRNNLM, self).__init__()
    self.bidirectional = True
    self.dropout = 0.5
    self.vocab_size = vocab_size
    self.word_emb = nn.Embedding(vocab_size, 48)  # (seq_len, batch_size, 48)
    self.rnn = nn.LSTM(input_size=48, hidden_size=32, num_layers=3,
        dropout=self.dropout, bidirectional=self.bidirectional)  # (seq_len, batch_size, 32 * (2 or 1))
    self.final = nn.Linear(32 * (2 if self.bidirectional else 1), vocab_size)  # (seq_len, batch_size, vocab_size)
    self.softmax = nn.Softmax()
    self.h0 = nn.Parameter(torch.randn(1, 32))
    self.c0 = nn.Parameter(torch.randn(1, 32))

  def forward(self, input_batch):
    batch_size = input_batch.size()[1]
    word_emb = self.word_emb(input_batch)
    h0 = self.h0.unsqueeze(1).expand(3 * (2 if self.bidirectional else 1), batch_size, 32).contiguous()
    c0 = self.c0.unsqueeze(1).expand(3 * (2 if self.bidirectional else 1), batch_size, 32).contiguous()
    hidden, _ = self.rnn(word_emb, (h0, c0))
    output_batch = torch.log(self.softmax(self.final(hidden).view(-1, self.vocab_size)))
    return output_batch



logging.basicConfig(
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S', level=logging.INFO)

parser = argparse.ArgumentParser(description="test code.")
parser.add_argument("--bidirectional", default=True, type=lambda x: (str(x).lower() == 'true'),
                    help="if use bidirectional LSTM.")
parser.add_argument("--dropout", default=0.5, type=float,
                    help="dropout value for LSTM.")
parser.add_argument("--gpuid", default=[], nargs='+', type=int,
                    help="ID of gpu device to use. Empty implies cpu usage.")


def main(options):

    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False
    random.seed(1)
    np.random.seed(1)
    os.environ['PYTHONHASHSEED'] = str(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.benchmark = False

    use_cuda = (len(options.gpuid) >= 1)
    if options.gpuid:
        torch.cuda.set_device(options.gpuid[0])

    vocab_size = 100
    rnnlm = BiRNNLM(vocab_size, bidirectional=options.bidirectional,
        dropout=options.dropout)
    if use_cuda > 0:
        rnnlm.cuda()
    else:
        rnnlm.cpu()

    criterion = nn.NLLLoss()
    optimizer = torch.optim.Adam(rnnlm.parameters(), 0.001)

    batch_size = 64
    seq_len = 100
    for i in range(1000):
        train_batch = torch.randint(vocab_size, (seq_len, batch_size), dtype=torch.int64)
        if use_cuda:
            train_batch = train_batch.cuda()

        sys_out_batch = rnnlm(train_batch)  # (seq_len, batch_size, vocab_size)
        sys_out_batch = sys_out_batch.view(-1, vocab_size)
        train_out_batch = train_batch.view(-1)
        loss = criterion(sys_out_batch, train_out_batch)
        print("loss at batch {0}: {1}".format(i, loss.data.item()))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


if __name__ == "__main__":
    ret = parser.parse_known_args()
    options = ret[0]
    if ret[1]:
        logging.warning("unknown arguments: {0}".format(parser.parse_known_args()[1]))
    main(options)