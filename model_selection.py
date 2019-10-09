import os
import argparse
import numpy as np
import pandas as pd
import shutil
parser = argparse.ArgumentParser(description='BERT fine_tune model selection(grid search)')
parser.add_argument('-lr-low', type=float, default=2e-5, help='minimum value of learning rate [default:2e-5]')
parser.add_argument('-lr-high', type=float, default=2e-4, help='maximum value of learning rate [default:2e-4]')
parser.add_argument('-lr-step', type=float, default=5e-6, help='step size of learning rate [default:5e-6]')
parser.add_argument('-epoch-low', type=float, default=2, help='minimum number of epochs [default:2]')
parser.add_argument('-epoch-high', type=float, default=6, help='maximum number of epochs [default:6]')
parser.add_argument('-epoch-step', type=float, default=1, help='step size of epochs [default:1]')
parser.add_argument('-batch-min', type=int, default=2, help='minimum batch size [default:2]')
parser.add_argument('-batch-max', type=int, default=48, help='maximum batch size [default:2]')
parser.add_argument('-batch_step', type=int, default=4, help='step size for batch size setups [default:4]')
# seq
parser.add_argument('-seqlen-low', type=int, default=16, help='minimum value of sequence length')
parser.add_argument('-validation_split', type=int, default=4, help='K of k-fold validation [default:4]')
parser.add_argument('-task', type=str, default='vp', help='task for the classification [default:vp]')
parser.add_argument('-simple-hold-out', action='store_true', default=False,
                    help='if set to true, use low parameters for a single evaluation instead of k-fold')
parser.add_argument('-base-model', type=str, default='uncased_L-12_H-768_A-12',
                    help='base model [default:uncased_L-12_H-768_A-12]')
parser.add_argument('-initial-path', type=str, default='./uncased_L-12_H-768_A-12/bert_model.ckpt')
parser.add_argument('-output-dir', type=str, default='./tmp/vp_val')
parser.add_argument('-srun', action='store_true', default=False, help='set if to use srun[default:False]')
parser.add_argument('-server', default='vibranium',
                    help='set server for slurm, if srun not set to True, ignored. [default:bibranium]')
parser.add_argument('-gpu', type=int, default=0,
                    help='gpu device for slurm, if srun is not set to True, ignored. [default:1]')
args = parser.parse_args()
DATA_BASE_PATH = './data'
project_data_path = os.path.join(DATA_BASE_PATH, args.task)
ALL_FILE_NAME='all.tsv'
VAL_DIR = 'validation'
if ALL_FILE_NAME not in os.listdir(project_data_path):
    # no "all tsv" exists
    raise NotImplementedError
all_data = os.path.join(project_data_path, ALL_FILE_NAME)
df = pd.read_csv(all_data,sep='\t', header=None, names=['labels', 'text'])
shuffled_list = np.array_split(df, args.validation_split)
if VAL_DIR not in os.listdir(project_data_path):
    os.mkdir(os.path.join(project_data_path,VAL_DIR))
validation_dirs = []
for i in range(args.validation_split):
    validation_dir = os.path.join(project_data_path, VAL_DIR, VAL_DIR + str(i))
    if VAL_DIR+str(i) not in os.listdir(os.path.join(project_data_path,VAL_DIR)):
        os.mkdir(validation_dir)
    validation_dirs.append(validation_dir)
    shuffled_list[i].to_csv(os.path.join(validation_dir,'dev.tsv'), header=False, sep='\t',index=False)
    train = pd.concat(shuffled_list[:i]+shuffled_list[i+1:])
    train.to_csv(os.path.join(validation_dir,'train.tsv'), header=False, sep='\t',index=False)

# TODO
# def generate_command(lr, batch, epoch, datapath):

#     command=""
#     if args.srun:
#         command += 'srun' +

def fun():
    os.system("""
python run_classifier.py --task_name=vp --do_train=true --do_eval=true --data_dir=./data/vp --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/vp_output
""")


# for i in range(4):
#     fun()
# os.system("""rm -rf """+os.path.join(project_data_path,VAL_DIR))

shutil.rmtree(os.path.join(project_data_path,VAL_DIR))