import torch, numpy as np, time
from torch.autograd import Variable

from model.model import make_model

from training.decode import greedy_decode
from training.opt import NoamOpt
from training.label_smooth import LabelSmoothing
from training.batch import Batch, subsequent_mask
from utils import attention_visualization

# Synthetic Data
def data_gen(V, batch, nbatches, device):
  """
  Generate random data for a src-tgt copy task.
  Data generated is of batch_size `batch`, sequence_length 10 of random integers from 1 to `V`.
  The data matrix is thus size (batch, 10). `V` represents the vocab size.
  The first token is set to 1 ("the sentence start token").
  Both the src and the tgt sequences are set to the same things, which means that the transformer learns to predict the
  same sequence. (This is a commonly used first test-case).
  """
  for i in range(nbatches):
      data = torch.from_numpy(np.random.randint(1, V, size=(batch, 10)))    # Matrix shpe (batch, 10) filled with randint between 1 and V-1
      data[:, 0] = 1
      src = Variable(data, requires_grad = False).to(device)
      tgt = Variable(data, requires_grad = False).to(device)
      print(data)
      yield Batch(src, tgt, 0)

# Loss Computation
class SimpleLossCompute:
    """
    A simple loss compute and train function. We decode the output text one char position at a time.
    """
    def __init__(self, generator, criterion, opt=None):
        self.generator = generator
        self.criterion = criterion
        self.opt = opt

    def __call__(self, out, y, norm):
        total = 0.0
        out_grad = []
        for i in range(out.size(1)):
            out_column = Variable(out[:, i].data, requires_grad=True)
            gen = self.generator(out_column)
            loss = self.criterion(gen, y) / norm
            total += loss.item()
            loss.backward()
            out_grad.append(out_column.grad.data.clone())
        out_grad = torch.stack(out_grad, dim=1)
        out.backward(gradient=out_grad)

        if self.opt is not None:
            self.opt.step()
            self.opt.optimizer.zero_grad()
            # Note: Since the backward() function accumulates gradients, and you don’t want to mix up gradients between minibatches,
            # you have to zero them out at the start of a new minibatch. This is exactly like how a general (additive) accumulator
            # variable is initialized to 0 in code.
        return total


def run_epoch(data_iter, model, loss_compute):
    "Standard Training and Logging Function"
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    for i, batch in enumerate(data_iter):
        out = model.forward(batch.src, batch.trg, batch.src_mask, batch.trg_mask)
        loss = loss_compute(out, batch.trg_y, batch.ntokens)
        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % 50 == 1:
            elapsed = time.time() - start
            print("Epoch Step: %d Loss: %f Tokens per Sec: %f" %
                  (i, loss / batch.ntokens, tokens / elapsed))
            start = time.time()
            tokens = 0
    return total_loss / total_tokens


# Greedy Decoding
# Train the simple copy task
def train_simple_copy(device='cpu'):
    """
    `V` is vocab size for source and target.
    We make a Transformer transformer with N=2 replications at each layer,
    train the transformer using randomly generated data from `data_gen()`
    and test the data on further random data from `data_gen()`.
    Since `data_gen()` produces src and tgt sequences which are the same,
    the transformer learns to predict the same sequence.

    Explaination: `run_epoch` runs the epoch.

    `SimpleLossCompute` is simply a customizeable function where we can control how the loss is computed.
    It starts from the final embeddings in the transformer, uses the transformer generator to generate the next word
    and computes the loss. In this case the loss function we use is LabelSmoothing + KLDivLoss (inside of LabelSmoothing class).
    """
    V = 11
    criterion = LabelSmoothing(size=V, padding_idx=0, smoothing=0.0)
    criterion = criterion.to(device)
    model = make_model(V,V,N=2)
    model = model.to(device)
    model_opt = NoamOpt(model.src_embed[0].d_model, 1, 400,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    for epoch in range(50):
        # In pytorch, training and eval are performed my instantiating the transformer, calling `transformer.train()`
        # Then calling `transformer.eval()`.
        model.train()
        run_epoch(data_gen(V, 30, 20, device), model,
                      SimpleLossCompute(model.generator, criterion, model_opt))
        model.eval()
        print(run_epoch(data_gen(V, 30, 5, device), model,
                        SimpleLossCompute(model.generator, criterion, None)))
    return model


def _run_simple_copy_train_decode():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = train_simple_copy(device)
    model.eval()
    src = Variable(torch.LongTensor([[1,2,3,4,5,6,7,8,9,10]])).to(device)
    src_mask = Variable(torch.ones(1,1,10)).to(device)
    trans = greedy_decode(model, src, src_mask, max_len=10, start_symbol=1, device=device)
    src = src.cpu().numpy().squeeze(0)
    trans = trans.cpu().numpy().squeeze(0)
    attention_visualization(model, trans, src)
