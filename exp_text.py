#! /usr/bin/env python
import os
import argparse
import datetime
import torch
from torchtext.legacy import data
import numpy as np
import model
import train
import mydatasets
import matplotlib.pyplot as plt
os.environ['CUDA_VISIBLE_DEVICES'] = '3'

parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.01, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=50, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=50, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-predict_and_attack', type=str, default=None, help='predict and attack the sentence given')
parser.add_argument('-test_and_attack', action='store_true', default=False, help='eval and attack on the entire dataset')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-example_save_path', type=str, default=False, help='where to saves example images for explainations')

args = parser.parse_args()

def normalize_words(abs_attack): 
    """normalize saliency across embd index"""
    average_attack = np.mean(abs_attack,axis = 0) #average saliency for each embd index
    normalized_attack_sq = (abs_attack-average_attack)**2
    return normalized_attack_sq

# load SST dataset
def sst(text_field, label_field,  **kargs):
    train_data, dev_data, test_data = datasets.SST.splits(text_field, label_field, fine_grained=True)
    text_field.build_vocab(train_data, dev_data, test_data)
    label_field.build_vocab(train_data, dev_data, test_data)
    train_iter, dev_iter, test_iter = data.BucketIterator.splits(
                                        (train_data, dev_data, test_data), 
                                        batch_sizes=(args.batch_size, 
                                                     len(dev_data), 
                                                     len(test_data)),
                                        **kargs)
    return train_iter, dev_iter, test_iter 

# load MR dataset
def mr(text_field, label_field, **kargs):
    train_data, dev_data = mydatasets.MR.splits(text_field, label_field)
    text_field.build_vocab(train_data, dev_data)
    label_field.build_vocab(train_data, dev_data)
    if args.predict_and_attack:
        train_iter=None
        dev_iter=None     
    else:   
        train_iter, dev_iter = data.Iterator.splits(
                                    (train_data, dev_data), 
                                    batch_sizes=(args.batch_size, len(dev_data)),
                                    **kargs)
    return train_iter, dev_iter


# load data
text_field = data.Field(lower=True)
label_field = data.Field(sequential=False)
train_iter, dev_iter = mr(text_field, label_field, device=-1, repeat=False)

# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    # print('\nLoading model from {}...'.format(args.snapshot))
    if args.cuda:
        cnn.load_state_dict(torch.load(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot, map_location=lambda storage, loc: storage))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
elif args.predict_and_attack:
    label = train.predict_and_attack(args.predict_and_attack, cnn, text_field, label_field, args.cuda, args.example_save_path)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict_and_attack, label))

# Plot explanations
clean_symbols=True
plot_averaged_saliency=True
x_index = np.load('tmp_saved_data/x_index.npy',allow_pickle=True)
pred_adv_lst = np.load('tmp_saved_data/pred_adv_lst.npy',allow_pickle=True)
attacks_lst_PGDM = np.load('tmp_saved_data/attacks_lst_PGDM.npy',allow_pickle=True)
text = np.load('tmp_saved_data/text.npy',allow_pickle=True)
abs_attack_pgdm = np.absolute(np.sum(attacks_lst_PGDM,axis=0)[x_index])
abs_attack_fgm = np.absolute(attacks_lst_PGDM[0][x_index])

normalized_score = normalize_words(abs_attack_fgm).sum(axis=1).reshape(-1,1)
text_with_score = [text[ii]+"_{}".format(normalized_score[ii]) for ii in range(len(text))]

if clean_symbols:
    clean_ind = np.invert((text==',')+(text=='')+(text=='.'))
else:
    clean_ind = np.ones((len(text),), dtype=bool)


# plot saliency map
fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
im = ax.imshow(normalize_words(abs_attack_fgm),cmap='hot')
ax.set_yticks(np.arange(len(text)))
ax.set_yticklabels(text)
ax.set_xlabel ('Embedding Dimension')
plt.savefig('results/exp_txt_sample_embedding.png')

# plot averaged saliency
fig = plt.figure()
ax = fig.add_subplot()
fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=.97)
im = ax.imshow(normalized_score[clean_ind],cmap='hot')
ax.set_yticks(np.arange(len(text[clean_ind])))
ax.set_yticklabels(np.array(text_with_score)[clean_ind])
plt.xticks([])
plt.savefig('results/exp_txt_sample_averaged.png')
