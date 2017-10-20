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

'''
make model
'''

def make_model(hidden_dim, wordvec_dim, vocab_size):
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

    def forward(features, captions):
        loss = Variable(torch.Tensor([0.0]))

        for feature, caption in zip(features, captions):
            print(len(caption))
            feature_var = Variable(torch.from_numpy(np.array([feature])))
            # print(feature_var.data.shape)
            # caption_var = Variable(torch.from_numpy(np.array(captions)))
            h = proj(feature_var) # torch.mm(feature_var, W_proj) + b_proj
            # print(h.data.shape)
            c = Variable(torch.from_numpy(np.zeros_like(h.data.numpy())))
            a = attention(h)
            a = softmax(a)
            z = a*feature_var

            for i in range(len(caption)-1):
                input_word = embeds(Variable(torch.LongTensor([int(caption[i])]))) # W_embed[caption[i]]
                target_var = Variable(torch.LongTensor([int(caption[i+1])]))
                input_cell = ztrans(z) + input_word
                h, c = cell(input_cell, (h, c))

                scores = vocab(h)
                loss += loss_fn(scores, target_var)
                a = attention(h)
                a = softmax(a)
                z = a*feature_var
        return loss
    return params, forward

def step(loss, optimizer):
    loss /= batch_size
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

'''
training
'''

data = load_coco_data()
print(data['train_features'].shape)
print(data['val_features'].shape)

num_epochs = 100
batch_size = 25
num_train, feature_dim = data['train_features'].shape
hidden_dim = 512
wordvec_dim = 256
vocab_size = len(data['word_to_idx'])
batch_size = 25

iterations_per_epoch = max(num_train // batch_size, 1)
num_iterations = num_epochs * iterations_per_epoch

params, model = make_model(hidden_dim, wordvec_dim, vocab_size)
optimizer = optim.Adam(params, lr=1e-2)

for t in range(num_iterations):
    print(t, num_iterations)
    minibatch = sample_coco_minibatch(data,
            batch_size=batch_size,
            split='train')
    captions, features, urls = minibatch
    loss = model(features, captions)
    print(loss)
    step(loss, optimizer)

