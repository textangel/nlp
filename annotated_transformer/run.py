from docopt import docopt
import torch
import numpy as np
import time

import torch.nn as nn
from model.model import make_model
from vocab import Vocab
from training.opt import NoamOpt
from torch.autograd import Variable
from preprocess import batch_iter, pad_sents, read_corpus

def read_corpus(file_path, source):
    """ Read file, where each sentence is dilineated by a `\n`.
    @param file_path (str): path to file containing corpus
    @param source (str): "tgt" or "src" indicating whether text
        is of the source language or target language
    """
    data = []
    for line in open(file_path):
        sent = line.strip().split(' ')
        # only append <s> and </s> to the target sentence
        if source == 'tgt':
            sent = ['<s>'] + sent + ['</s>'] #TODO: Change
        data.append(sent)
    return data


 def __call__(self, out, y, norm):
        total = 0.0
        out_grad = []
        for i in range(out.size(1)):
            out_column = Variable(out[:, i].data, requires_grad=True)
            gen = self.generator(out_column)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(gen, y) / norm
            total += loss.item()
            loss.backward()
            out_grad.append(out_column.grad.data.clone())
        out_grad = torch.stack(out_grad, dim=1)
        out.backward(gradient=out_grad)



def compute_model_loss(model, generator, out, y, norm):
    total = 0.0
    out_grad = []
    for i in range(out.size(1)):
        out_column = Variable(out[:, i].data, requires_grad=True)
        gen = generator(out_column)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(gen, y) / norm
        total += loss.item()
        loss.backward()
        out_grad.append(out_column.grad.data.clone())
    out_grad = torch.stack(out_grad, dim=1)
    out.backward(gradient=out_grad)


def train(args: Dict):
    """ Train the NMT Model.
    @param args (Dict): args from cmd line
    :param args:
    """
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print('use device: %s' % device)

    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')

    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')

    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))

    train_batch_size = int(args['--batch-size'])
    N = int(args['--N'])
    d_model = int(args['--d_model'])
    d_ff = int(args['--d_ff'])
    h = int(args['--h'])
    dropout = float(args['--dropout'])

    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    lr=float(args['--lr'])

    vocab = Vocab.load(args['--vocab'])
    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    model = make_model(len(vocab.src), len(vocab.tgt), N, d_model, d_ff, h, dropout)
    model = model.to(device)

    optimizer = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.98), eps=1e-9))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_exmaples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood Training')

    while True:
        epoch += 1
        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            optimizer.zero_grad()
            batch_size = len(src_sents)

            example_losses = - model(src_sents, tgt_sents) #(batch_size,)
            batch_loss = example_losses.sum()




def main():
    """ Main func.
    """
    args = docopt(__doc__)

    # seed the random number generators
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        raise (NotImplementedError)
        # TODO add
    else:
        raise RuntimeError('invalid run mode')

if __name__ == '__main__':
    main()