import pandas as pd
import tensorflow as tf
import os
import argparse
import pathlib
import numpy as np
from parse_bert_embedding_json import *
"""
current version: even if we have only a single layer to extract, if we use concat4, it will still work correctly.
even if we have layers other than 4, it will still concatenate it correctly.
"""
parser = argparse.ArgumentParser(description='BERT feature extraction for vp dataset')
parser.add_argument('-method', type=str, default='concat4',
                    help="""Strategy for extract  features
                    concat4 represent concat the output of last four layers of bert
                    [default:concat4]""")
parser.add_argument('-bert-model', type=str, default='./uncased_L-12_H-768_A-12',
                    help="""Directory for bert model config file and vocab list
                    [default:./uncased_L-12_H-768_A-12""")
parser.add_argument('-input', type=str, default='./data/vp/',
                    help="""Directory for input file, script will search for all.tsv and labels.txt in this dir
                    [default:./data/vp/]""")
parser.add_argument('-temp-dir', type=str, default='./tmp/vp_extract',
                    help="""temporary directory for holding intermidiate files
                    [default:./tmp/vp_extract]""")
parser.add_argument('-output-dir', type=str, default='./data/vp/bert_embeddings',
                    help="""output dir of extracted feature
                    [default:./data/vp/bert_embeddings]""")
parser.add_argument('-sequence-length', type=int, default=64,
                    help='padded sequence length [default:64]')
parser.add_argument('-checkpoint-name', type=str, default='bert_model.ckpt',
                    help="""bert checkpoint name, under the bert model directory
                    [default: bert_model.ckpt]""")
parser.add_argument('-batch-size', type=int, default=8,
                    help="""batch size [default:8]""")
parser.add_argument('-layers', type=str, default='-1,-2,-3,-4,-5,-6,-7,-8,-9,-10,-11,-12', help="extracted layer")

# arguments and variables
args = parser.parse_args()
pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
pathlib.Path(args.temp_dir).mkdir(parents=True, exist_ok=True)
training_data_path = os.path.join(args.input,'all.tsv')
labels_data_path = os.path.join(args.input, 'labels.txt')

TRAIN_TEXT_NAME = os.path.join(args.temp_dir, 'train_text.txt')
LABEL_TEXT_NAME = os.path.join(args.temp_dir, 'label_text.txt')

TRAIN_FEATURE_NAME = os.path.join(args.temp_dir, 'train_feature.json')
LABEL_FEATURE_NAME = os.path.join(args.temp_dir, 'label_feature.json')

EMBEDDED_TRAIN_DATA = os.path.join(args.output_dir, 'all.npy')
EMBEDDED_LABEL_DATA = os.path.join(args.output_dir, 'labels.npy')

VOCAB_ = os.path.join(args.bert_model, 'vocab.txt')
BERT_CONFIG = os.path.join(args.bert_model, 'bert_config.json')
CHECKPOINT = os.path.join(args.bert_model, args.checkpoint_name)


LAYERS = args.layers
# read data
df_train = None
df_labels = None
try:
    df_train = pd.read_csv(training_data_path, sep='\t', header=None, names=['labels','text'])
    df_labels = pd.read_csv(labels_data_path, sep='\t', header=None, names=['labels','text'])
except None:
    exit()
# extract text
# get the text column of the data and store it into target txt file
text_train = df_train['text']
text_train = text_train.to_frame()
text_train.to_csv(TRAIN_TEXT_NAME, header=False, sep='\t', index=False)

text_labels = df_labels['text']
text_labels = text_labels.to_frame()

text_labels.to_csv(LABEL_TEXT_NAME,header=False, sep='\t', index=False)

def generate_command(input_dir, output_dir, vocab_file, bert_config, init_checkpoint, layers, max_seq_length, batch_size):
    """
    generete command
    :param input_dir:
    :param output_dir:
    :param vocab_file:
    :param bert_config:
    :param init_checkpoint:
    :param layers:
    :param max_seq_length:
    :param batch_size:
    :return:
    """
    result = ""
    result += "python extract_features.py "
    result += "--input_file="+input_dir
    result += " --output_file=" + output_dir
    result += " --vocab_file="+vocab_file
    result += " --bert_config_file="+bert_config
    result += " --init_checkpoint="+init_checkpoint
    result += " --layers="+layers
    result += " --max_seq_length="+str(max_seq_length)
    result += " --batch_size="+str(batch_size)
    return result


command_train = generate_command(input_dir=TRAIN_TEXT_NAME, output_dir=TRAIN_FEATURE_NAME,vocab_file=VOCAB_,
                                 bert_config=BERT_CONFIG, init_checkpoint=CHECKPOINT, layers=LAYERS,
                                 max_seq_length=args.sequence_length,batch_size=args.batch_size)
os.system(command_train)
command_label = generate_command(input_dir=LABEL_TEXT_NAME, output_dir=LABEL_FEATURE_NAME, vocab_file=VOCAB_,
                                 bert_config=BERT_CONFIG, init_checkpoint=CHECKPOINT, layers=LAYERS,
                                 max_seq_length=args.sequence_length, batch_size=args.batch_size)
os.system(command_label)


feature_train = parse_bert_embedding_json(TRAIN_FEATURE_NAME, args.method)
# df_train_embed = df_train['labels'].to_frame()
# df_train_embed['text'] = feature_train
# df_train_embed.to_csv(EMBEDDED_TRAIN_DATA, header=False, sep='\t', index=False)
np.save(EMBEDDED_TRAIN_DATA, feature_train)

feature_label = parse_bert_embedding_json(LABEL_FEATURE_NAME, args.method)
# df_labels_embed = df_labels['labels'].to_frame()
# df_labels_embed['text'] = feature_label
# print(type(feature_label[0]))
# print(df_train_embed['text'].dtype)
# df_labels_embed.to_csv(EMBEDDED_LABEL_DATA, header=False, sep='\t', index=False)
np.save(EMBEDDED_LABEL_DATA, feature_label)