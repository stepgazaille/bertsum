# -*- coding: utf-8 -*-
import argparse
import os
import torch

# constants
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'

PAD_TOKEN = '[PAD]' # This has a vocab id, which is used to pad the encoder input, decoder input and target sequence
UNKNOWN_TOKEN = '[UNK]' # This has a vocab id, which is used to represent out-of-vocabulary words
START_DECODING = '[START]' # This has a vocab id, which is used at the start of every decoder input sequence
STOP_DECODING = '[STOP]' # This has a vocab id, which is used at the end of untruncated target sequences
BERT_START = '[CLS]'
seq_len = 100

parser = argparse.ArgumentParser()
parser.add_argument("--persona", action="store_true")
parser.add_argument("--hidden_dim", type=int, default=3072)
parser.add_argument("--emb_dim", type=int, default=768)
parser.add_argument("--batch_size", type=int, default=36)
parser.add_argument("--lr", type=float, default=0.0003)
parser.add_argument("--max_grad_norm", type=float, default=2.0)
parser.add_argument("--max_enc_steps", type=int, default=400)
parser.add_argument("--max_dec_steps", type=int, default=seq_len) #100
parser.add_argument("--min_dec_steps", type=int, default=0) #35
parser.add_argument("--beam_size", type=int, default=5)
parser.add_argument("--train_data_path", type=str, default="/home/bertsum/data/finished_files")
parser.add_argument("--save_path", type=str, default="save/")
parser.add_argument("--save_path_dataset", type=str, default="save/")
parser.add_argument("--cuda", action="store_true")
parser.add_argument("--pointer_gen", action="store_true")
parser.add_argument("--is_coverage", action="store_true")
parser.add_argument("--use_oov_emb", action="store_true")
parser.add_argument("--pretrain_emb", action="store_true")
parser.add_argument("--test", action="store_true")
parser.add_argument("--model", type=str, default="trs")
parser.add_argument("--label_smoothing", action="store_true")
parser.add_argument("--noam", action="store_true")
parser.add_argument("--split_copy_head", action="store_true")
parser.add_argument("--small", action="store_true", help="using smaller data. suitable for debug/testing")
parser.add_argument("--nb_epochs", type=int, default=1, help="max number of epochs") # 4

## transformer 
parser.add_argument("--hop", type=int, default=12)
parser.add_argument("--heads", type=int, default=12) 
parser.add_argument("--depth", type=int, default=48)
parser.add_argument("--filter", type=int, default=50)

## BERT
parser.add_argument("--max_seq_length", default=512, type=int,
                        help="The maximum total input sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")

parser.add_argument("--max_tgt_length", default=100, type=int,
                        help="The maximum total target sequence length after WordPiece tokenization. Sequences longer "
                            "than this will be truncated, and sequences shorter than this will be padded.")

arg = parser.parse_args()
print(arg)
model = arg.model
persona = arg.persona

# Hyperparameters
hidden_dim= arg.hidden_dim
emb_dim= arg.emb_dim
batch_size= arg.batch_size
lr=arg.lr

max_enc_steps = arg.max_enc_steps
max_dec_step = max_dec_steps = arg.max_dec_steps

min_dec_steps=arg.min_dec_steps 
beam_size=arg.beam_size

adagrad_init_acc=0.1
rand_unif_init_mag=0.02
trunc_norm_init_std=1e-4
max_grad_norm=arg.max_grad_norm

USE_CUDA = arg.cuda and torch.cuda.is_available()
pointer_gen = arg.pointer_gen
split_copy_head = arg.split_copy_head
is_coverage = arg.is_coverage
use_oov_emb = arg.use_oov_emb
cov_loss_wt = 1.0
lr_coverage=0.15
eps = 1e-12
epochs = arg.nb_epochs
UNK_idx = 0
PAD_idx = 1
EOS_idx = 2
SOS_idx = 3
SENTINEL = 4

save_path = arg.save_path
if not os.path.exists(save_path):
    os.makedirs(save_path)

test = arg.test
if(test):
    save_path_dataset = arg.save_path_dataset
    if not os.path.exists(save_path_dataset):
        os.makedirs(save_path_dataset)
else:
    save_path_dataset = save_path


### transformer 
hop = arg.hop # layer #
heads = arg.heads # head #
depth = arg.depth # total_key_depth, total_value_depth
filter = arg.filter # filter_size

label_smoothing = arg.label_smoothing
noam = arg.noam

### BERT
train_data_path = os.path.expanduser(arg.train_data_path)
max_seq_length = arg.max_seq_length # initially was 128, but i changed to 512
max_tgt_length = arg.max_tgt_length
vocab_size = 30522
small = arg.small
