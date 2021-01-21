import argparse
import pprint

parser = argparse.ArgumentParser(description='argument parser')

# Misc
parser.add_argument('--ckpt_dir', type=str, default='./checkpoints/',
                        help='path for saving trained models')
parser.add_argument('--ckpt_path', type=str, default='./checkpoints/scacnn-model-10.pkl',
                        help='path for checkpont of trained models')
parser.add_argument('--result_dir', type=str, default='./results',
                    help='path for test output json file')
parser.add_argument('--gpu', type=str, default='0,1',
                    help='gpu ids to use')
parser.add_argument('--crop_size', type=int, default=224,
                    help='size for randomly cropping images')
parser.add_argument('--save_step', type=int, default=1,
                    help='num epochs for saving trained models')
parser.add_argument('--log_file', type=str, default='log.txt',
                    help='path for test output json file')
parser.add_argument('--beam_width', type=int, default=1,
                    help='beam width of beam search')
# flags
parser.add_argument('--restore_train', action='store_true',
                    help='set this flag to restore training from previous checkponts')
parser.add_argument('--fine_tune', action='store_true',
                    help='set this flag to fine-tune ImageNet model')



# Optimizer
parser.add_argument('--num_epochs', type=int, default=100,
                    help='number of total epochs')
parser.add_argument('--batch_size', type=int, default=20,
                    help='number of batch size')
parser.add_argument('--learning_rate', type=float, default=0.001,
                    help='learning rate for optimizer')

# Data
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers for data loader')
parser.add_argument('--vocab_path', type=str, default='./data/vocab.pkl',
                    help='path for vocabulary wrapper')
parser.add_argument('--train_image_dir', type=str, default='./data/train2014',
                    help='directory for train resized images')
parser.add_argument('--train_caption_path', type=str,
                    default='./data/annotations/captions_train2014.json',
                    help='path for train annotation json file')
parser.add_argument('--val_image_dir', type=str, default='./data/val2014',
                    help='directory for val resized images')
parser.add_argument('--val_caption_path', type=str,
                    default='./data/annotations/captions_val2014.json',
                    help='path for val annotation json file')


# Model
parser.add_argument('--model', type=str, default='nic', choices=['nic', 'ssa', 'scacnn'],
                    help='name for model')
parser.add_argument('--att_mode', type=str, default='cs', choices=['cs', 'sc', 'c', 's'],
                    help='attention mode for scacnn')
parser.add_argument('--embed_size', type=int, default=100,
                    help='dimension of word embedding vectors')
parser.add_argument('--hidden_size', type=int, default=300,
                    help='dimension of lstm hidden states')
parser.add_argument('--num_layers', type=int, default=1,
                    help='number of layers in lstm')
parser.add_argument('--dropout', type=float, default=0.1,
                    help='dropout rate for lstm')


def print_args(args):
    pprint.pprint(vars(args))
