import os
import argparse
import numpy as np
import pandas as pd
import shutil
import tensorflow as tf

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
parser.add_argument('-seqlen-low', type=int, default=16, help='minimum value of sequence length [default:16]')
parser.add_argument('-seqlen-high', type=int, default=64, help='maximum value of sequence length [default:64]')
parser.add_argument('-seqlen_step', type=int, default=2, help='step size for sequence length trials [default:2]')

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
parser.add_argument('-clean',action='store_true', default=False, help='set if to clean the files[default:False]')


args = parser.parse_args()
DATA_BASE_PATH = './data'
project_data_path = os.path.join(DATA_BASE_PATH, args.task)
ALL_FILE_NAME = 'all.tsv'
VAL_DIR = 'validation'
try:
    os.mkdir(args.output_dir)
except Exception:
    pass
if ALL_FILE_NAME not in os.listdir(project_data_path):
    # no "all tsv" exists
    raise NotImplementedError
all_data = os.path.join(project_data_path, ALL_FILE_NAME)
df = pd.read_csv(all_data, sep='\t', header=None, names=['labels', 'text'])
sup_df = pd.read_csv(os.path.join(project_data_path,'labels.txt'),sep='\t', header=None, names=['labels','text'])
shuffled_list = np.array_split(df, args.validation_split)
if VAL_DIR in os.listdir(project_data_path):
    shutil.rmtree(os.path.join(project_data_path, VAL_DIR))
if VAL_DIR not in os.listdir(project_data_path):
    os.mkdir(os.path.join(project_data_path, VAL_DIR))
if args.output_dir in os.listdir('.'):
    shutil.rmtree(args.output_dir)
# validation dirs has the validation folds
validation_dirs = []
for i in range(args.validation_split):
    validation_dir = os.path.join(project_data_path, VAL_DIR, VAL_DIR + str(i))
    if VAL_DIR + str(i) not in os.listdir(os.path.join(project_data_path, VAL_DIR)):
        os.mkdir(validation_dir)
    validation_dirs.append(validation_dir)
    shuffled_list[i].to_csv(os.path.join(validation_dir, 'dev.tsv'), header=False, sep='\t', index=False)
    train = pd.concat(shuffled_list[:i] + shuffled_list[i + 1:] + [sup_df])
    train.to_csv(os.path.join(validation_dir, 'train.tsv'), header=False, sep='\t', index=False)

vocab_file =os.path.join(args.base_model, 'vocab.txt')
config_file = os.path.join(args.base_model, 'bert_config.json')
init_check = os.path.join(args.base_model, 'bert_model.ckpt')

def generate_command(max_seq_length, lr, batch, epoch, datapath, trial_identifier, output_dir):
    """

    :param lr:
    :param batch:
    :param epoch:
    :param datapath:
    :param trial_identifier: identifier to distinguish between different trials of running
    :return:
    """

    command = ""
    if args.srun:
        command += 'srun ' + '-u ' + '-w ' + args.server + ' ' + '--gres=gpu:' + str(args.gpu) + ' '
        command += '-J' + str(trial_identifier) + ' gpurun.sh ' + '-c 1 '

    command += 'python run_classifier.py --task_name='+args.task + ' --do_train=true --do_eval=true --data_dir='+datapath
    command += ' --vocab_file='+vocab_file + ' --bert_config_file='+config_file+' --init_checkpoint='+init_check
    command += ' --max_seq_length=' + str(max_seq_length) + ' --train_batch_size='+str(batch)+' --learning_rate='+str(lr)
    command += ' --num_train_epochs='+str(epoch) + ' --output_dir='+output_dir
    return command

def get_acc(output_dir):
    with open(os.path.join(output_dir, 'eval_results.txt')) as f:
        lines = f.readlines()
        words = lines[0].split(' ')
        return float(words[2])
def fun():
    os.system("""
python run_classifier.py --task_name=vp --do_train=true --do_eval=true --data_dir=./data/vp --vocab_file=./uncased_L-12_H-768_A-12/vocab.txt --bert_config_file=./uncased_L-12_H-768_A-12/bert_config.json --init_checkpoint=./uncased_L-12_H-768_A-12/bert_model.ckpt --max_seq_length=128 --train_batch_size=32 --learning_rate=2e-5 --num_train_epochs=3.0 --output_dir=./tmp/vp_output
""")
trial_id = 0
result = []
for max_seq_length in range(args.seqlen_low, args.seqlen_high+1, args.seqlen_step):
    for lr in np.arange(args.lr_low, args.lr_high+args.lr_step, args.lr_step):
        for batch_size in range(args.batch_min, args.batch_max+1, args.batch_step):
            for epoch in np.arange(args.epoch_low, args.epoch_high+args.epoch_step, + args.epoch_step):
                acc_sum = 0.0
                for data_path in validation_dirs:
                    current_trial_dir = os.path.join(args.output_dir, str(trial_id))
                    command = generate_command(max_seq_length,lr, batch_size, epoch,data_path,trial_id, current_trial_dir)
                    try:
                        os.system(command)
                        acc_sum += get_acc(current_trial_dir)
                    except tf.errors.ResourceExhaustedError:
                        acc_sum += -99999

                acc = acc_sum/float(len(validation_dirs))
                result.append([max_seq_length, lr, batch_size, epoch,acc])
                result = np.array(result)
                df = pd.DataFrame(data=result, columns=['max_seq_len', 'lr', 'batch_size', 'epoch', 'acc'])
                df.to_csv('./tmp/result.csv')
                trial_id+=1
            trial_id +=1
        trial_id +=1
    trial_id +=1
# result = np.array(result)

if args.clean:
    shutil.rmtree(os.path.join(project_data_path, VAL_DIR))
    shutil.rmtree(args.output_dir)
