from pathlib import Path
import os
import pickle
from tqdm import tqdm
import json
from pprint import pprint

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from utils.config import parser, print_args
from utils.logger import Logger
from utils.data_loader import get_loader
from models.ssa import SSA
from models.nic import NIC
from models.scacnn import SCACNN


args = parser.parse_args()
print_args(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

LogMaster = Logger()


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def main():
    logger = LogMaster.get_logger('eval')

    if not os.path.isfile(args.ckpt_path):
        print('checkpoint not found: ', args.ckpt_path)
        exit(-1)

    # Image preprocessing
    # For normalization, see https://github.com/pytorch/vision#models
    transform = transforms.Compose([
	transforms.Resize([256, 256]),
        transforms.RandomCrop(args.crop_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406),
                             (0.229, 0.224, 0.225))])

    # Load vocabulary wrapper.
    with open(args.vocab_path, 'rb') as f:
        vocab = pickle.load(f)

    logger.info('building data loader...')
    # Build data loader
    data_loader, image_ids = get_loader(args.val_image_dir, args.val_caption_path, vocab,
                                        transform, args.batch_size,
                                        shuffle=False, num_workers=args.num_workers, is_eval=True)

    logger.info('building model...')
    # Build the models
    vocab_size = len(vocab)

    if args.model == 'ssa':
        net = SSA(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size)
    elif args.model == 'nic':
        net = NIC(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size)
    elif args.model == 'scacnn':
        net = SCACNN(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size)
    else:
        net = None
        print('model name not found: ' + args.model)
        exit(-2)

    net.eval()

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            net = nn.DataParallel(net)
        net.cuda()

    net.zero_grad()
    logger.info('restoring pretrained model...')
    checkpoint = torch.load(args.ckpt_path)
    try:
        args_dict = checkpoint['args']
        args.batch_size = args_dict['batch_size']
        args.learning_rate = args_dict['learning_rate']
        args.att_mode = args_dict['att_mode']
        args.model = args_dict['model']
        args.embed_size = args_dict['embed_size']
        args.hidden_size = args_dict['hidden_size']
        args.num_layers = args_dict['num_layers']
        net.load_state_dict(checkpoint['net_state'])
        epoch = checkpoint['epoch']
        print('using loaded args from checkpoint:')
        pprint(args)
    except:
        net.load_state_dict(checkpoint)
        epoch = 0

    logger.info('start generating captions...')
    total_step = len(data_loader)
    start_token = vocab('<start>')
    end_token = vocab('<end>')
    syn_captions = []
    keys = {}
    for i, (images, inputs, targets, masks, lengths, img_ids) in tqdm(enumerate(data_loader), total=total_step, leave=False, ncols=80, unit='b'):
        images = to_var(images, requires_grad=False)
        if args.beam_width == 1:
            results = net.greedy_search(images, start_token).data.cpu().numpy()
        else:
            results = net.beam_search(images, start_token, beam_width=args.beam_width).data.cpu().numpy()

        results = list(results)  # each element is [seq_len, 1]

        for i in range(len(results)):
            sentence = ''
            res = list(results[i])
            img_id = img_ids[i]
            for w in res:
                if w == start_token:
                    continue
                elif w == end_token:
                    break
                word = vocab.idx2word[w]
                sentence += (' ' + word)
            # only keep one caption for each image
            try:
                _ = keys[img_id]
            except KeyError:
                keys[img_id] = 1
                item = {'image_id': img_id,
                        'caption': sentence}
                syn_captions.append(item)

    res_dir = Path(args.result_dir)
    if not res_dir.is_dir():
        res_dir.mkdir()
    result_path = res_dir / Path(args.model + '-' + str(epoch) + '-predictions.json')
    with open(str(result_path), 'w') as fout:
        json.dump(syn_captions, fout)
    logger.info(f'captions saved: {str(result_path)}')


if __name__ == '__main__':
    main()
