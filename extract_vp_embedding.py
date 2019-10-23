import pandas as pd
import tensorflow as tf
import os
import argparse

parser = argparse.ArgumentParser(description='BERT feature extraction for vp dataset')
parser.add_argument('-method', type=str, default='concat4',
                    help="""Strategy for extract  features
                    concat4 represent concat the output of last four layers of bert
                    [default:concat4]""")
parser.add_argument('-bert-model', type=str, default='./uncased_L-12_H-768_A-12',
                    help="""Directory for bert model config file and vocab list
                    [default:./uncased_L-12_H-768_A-12""")
parser.add_argument('-input', type=str, default='./data/vp/all.tsv',
                    help="""Directory for input file, should be tsv file for vp data
                    [default:./data/vp/all.tsv]""")
parser.add_argument('-temp-dir', type=str, default='./tmp/vp_extract',
                    help="""temporary directory for holding intermidiate files
                    [default:./tmp/vp_extract]""")
parser.add_argument('-output-dir', type=str, default='./data/vp/bert_embeddings',
                    help="""output dir of extracted feature
                    [default:./data/vp/vert_embeddings]""")
parser.add_argument('-sequence-length', type=int, default=64,
                    help='padded sequence length [default:64]')
