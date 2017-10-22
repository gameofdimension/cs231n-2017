from __future__ import print_function
import time, os, json
import numpy as np
import matplotlib.pyplot as plt

from cs231n.gradient_check import eval_numerical_gradient, eval_numerical_gradient_array
from cs231n.rnn_layers import *
from cs231n.captioning_solver import CaptioningSolver
from cs231n.classifiers.rnn import CaptioningRNN
from cs231n.coco_utils import load_coco_data, sample_coco_minibatch, decode_captions
from cs231n.image_utils import image_from_url

import torch
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchvision.transforms as T
import random
import nltk

'''
make model
'''

def make_model(hidden_dim, wordvec_dim, vocab_size, max_len=16):
    # W_embed = Variable(torch.randn(vocab_size, wordvec_dim)/100, requires_grad=True)
    embeds = torch.nn.Embedding(vocab_size, wordvec_dim)
    # W_proj = Variable(torch.randn(feature_dim, hidden_dim)/np.sqrt(feature_dim), requires_grad=True)
    # b_proj = Variable(torch.zeros(hidden_dim), requires_grad=True)
    proj = torch.nn.Linear(feature_dim, hidden_dim)
    # W_vocab = Variable(torch.randn(hidden_dim, vocab_size)/np.sqrt(hidden_dim), requires_grad=True)
    # b_vocab = Variable(torch.zeros(vocab_size), requires_grad=True)
    vocab = torch.nn.Linear(hidden_dim, vocab_size)
    # W_attention = Variable(torch.randn(hidden_dim, feature_dim)/np.sqrt(hidden_dim), requires_grad=True)
    # b_attention = Variable(torch.zeros(feature_dim), requires_grad=True)
    attention = torch.nn.Linear(hidden_dim, feature_dim)
    # W_z = Variable(torch.randn(feature_dim, wordvec_dim)/np.sqrt(feature_dim), requires_grad=True)
    ztrans = torch.nn.Linear(feature_dim, wordvec_dim)
    cell = torch.nn.LSTMCell(wordvec_dim, hidden_dim)
    softmax = torch.nn.Softmax()

    loss_fn = torch.nn.CrossEntropyLoss().type(torch.FloatTensor)
    params = list(embeds.parameters()) + list(proj.parameters()) + \
            list(vocab.parameters()) + list(attention.parameters()) + \
            list(ztrans.parameters()) + list(cell.parameters())
    # print(params)

    def sample(features, start, end, null, max_length=30):
        batch_size = features.data.shape[0]

        feature_var = Variable(torch.from_numpy(features))
        h = proj(feature_var)
        c = Variable(torch.from_numpy(np.zeros_like(h.data.numpy())))
        a = attention(h)
        a = softmax(a)
        z = a*feature_var
        captions = null * np.ones((batch_size, max_length), dtype=np.int32)

        # for k in range(batch_size):
        print("begin sample", time.time())
        h0 = h #[k:k+1, :]
        c0 = c #[k:k+1, :]
        a0 = a #[k:k+1, :]
        z0 = z #[k:k+1, :]
        step = 0
        current = [start]*batch_size
        while True:
            input_word = embeds(Variable(torch.LongTensor(np.array(current).tolist()))) # W_embed[caption[i]]
            input_cell = ztrans(z0) + input_word
            # print(input_word.data.shape, ztrans(z0).data.shape, input_cell.data.shape, h0.data.shape, c0.data.shape)
            h0, c0 = cell(input_cell, (h0, c0))

            scores = vocab(h0)
            a0 = attention(h0)
            a0 = softmax(a0)
            z0 = a0*feature_var
            # probs = softmax(scores)
            # batch_loss[:,i] = -torch.log(probs.gather(1, target_var.view(-1, 1)).squeeze())
            nextw = np.argmax(scores.data.numpy(), axis=1)
            # if nextw == end:
            #     break
            captions[:,step] = nextw
            step += 1
            if step >= max_length:
                break
            current = nextw
        for i in range(captions.shape[0]):
            end_flag = False
            for j in range(captions.shape[1]):
                if not end_flag:
                    if captions[i,j] == end:
                        end_flag = True
                else:
                    captions[i,j] = null
        print("finish sample", time.time())
        return captions

    def forward(features, captions):
        batch_size = features.data.shape[0]
        batch_loss = Variable(torch.from_numpy(np.zeros((batch_size, max_len)).astype(np.float32)))
        mask = Variable(torch.from_numpy((captions[:,1:] != 0).astype(np.float32)))

        feature_var = Variable(torch.from_numpy(features))
        h = proj(feature_var) # torch.mm(feature_var, W_proj) + b_proj
        c = Variable(torch.from_numpy(np.zeros_like(h.data.numpy())))
        a = attention(h)
        a = softmax(a)
        z = a*feature_var

        for i in range(max_len):
            input_word = embeds(Variable(torch.LongTensor(captions[:,i].tolist()))) # W_embed[caption[i]]
            target_var = Variable(torch.LongTensor(captions[:,i+1].tolist()))
            input_cell = ztrans(z) + input_word
            # print(input_cell.data.shape, h.data.shape, c.data.shape)
            h, c = cell(input_cell, (h, c))

            scores = vocab(h)
            probs = softmax(scores)
            batch_loss[:,i] = -torch.log(probs.gather(1, target_var.view(-1, 1)).squeeze())
            a = attention(h)
            a = softmax(a)
            z = a*feature_var

        batch_loss = batch_loss*mask
        loss = batch_loss.sum()
        return loss / batch_size
    return params, forward, sample

def step(loss, optimizer):
    # loss /= batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
training
'''

data = load_coco_data(max_train=10000)
print(data['train_features'].shape)
print(data['val_features'].shape)

num_epochs = 100
batch_size = 25
num_train, _ = data['train_captions'].shape
_, feature_dim = data['train_features'].shape
hidden_dim = 512
wordvec_dim = 256
vocab_size = len(data['word_to_idx'])
# print(vocab_size)
batch_size = 25

iterations_per_epoch = max(num_train // batch_size, 1)
num_iterations = num_epochs * iterations_per_epoch

params, model, sample = make_model(hidden_dim, wordvec_dim, vocab_size)
optimizer = optim.Adam(params, lr=1e-3)

for t in range(num_iterations):
    minibatch = sample_coco_minibatch(data,
            batch_size=batch_size,
            split='train')
    captions, features, urls = minibatch
    loss = model(features, captions)
    if t%10 == 0:
        print(time.strftime('%X %x %Z'), t, num_iterations, loss.data[0])
    step(loss, optimizer)

def demo(data, sample):
    start = data['word_to_idx']['<START>']
    end = data['word_to_idx']['<END>']
    null = data['word_to_idx']['<NULL>']

    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=2)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = sample(features, start, end, null)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            plt.imshow(image_from_url(url))
            plt.title('%s\n%s\nGT:%s' % (split, sample_caption, gt_caption))
            plt.axis('off')
            plt.show()

def BLEU_score(gt_caption, sample_caption):
    """
    gt_caption: string, ground-truth caption
    sample_caption: string, your model's predicted caption
    Returns unigram BLEU score.
    """
    reference = [x for x in gt_caption.split(' ') 
                 if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x and '<NULL>' not in x)]
    hypothesis = [x for x in sample_caption.split(' ') 
                  if ('<END>' not in x and '<START>' not in x and '<UNK>' not in x and '<NULL>' not in x)]
    BLEUscore = nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, weights = [1])
    return BLEUscore

def evaluate_model(data, sample):
    """
    model: CaptioningRNN model
    Prints unigram BLEU score averaged over 1000 training and val examples.
    """
    start = data['word_to_idx']['<START>']
    end = data['word_to_idx']['<END>']
    null = data['word_to_idx']['<NULL>']

    BLEUscores = {}
    for split in ['train', 'val']:
        minibatch = sample_coco_minibatch(data, split=split, batch_size=1000)
        gt_captions, features, urls = minibatch
        gt_captions = decode_captions(gt_captions, data['idx_to_word'])

        sample_captions = sample(features, start, end, null)
        sample_captions = decode_captions(sample_captions, data['idx_to_word'])

        total_score = 0.0
        for gt_caption, sample_caption, url in zip(gt_captions, sample_captions, urls):
            total_score += BLEU_score(gt_caption, sample_caption)

        BLEUscores[split] = total_score / len(sample_captions)

    for split in BLEUscores:
        print('Average BLEU score for %s: %f' % (split, BLEUscores[split]))

# demo(data, sample)
evaluate_model(data, sample)
