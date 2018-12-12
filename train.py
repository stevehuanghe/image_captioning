import os
import pickle
import pprint
import numpy as np
from tqdm import tqdm
from pathlib import Path

import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms

from utils.logger import Logger
from utils.config import parser, print_args
from utils.data_loader import get_loader
from models.ssa import SSA
from models.nic import NIC
from models.scacnn import SCACNN

args = parser.parse_args()
print_args(args)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

log_dir = Path('./logs')
if not log_dir.is_dir():
    log_dir.mkdir()
log_path = log_dir / Path(args.log_file)

LogMaster = Logger(log_path)


def to_var(x, requires_grad=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, requires_grad=requires_grad)


def main():
    logger = LogMaster.get_logger('main')

    # Create checkpoint directory
    if not os.path.exists(args.ckpt_dir):
        os.makedirs(args.ckpt_dir)

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

    if args.restore_train:
        if not os.path.isfile(args.ckpt_path):
            print('checkpoint not found: ', args.ckpt_path)
            exit(-1)
        checkpoint = torch.load(args.ckpt_path)
        args_dict = checkpoint['args']
        args.batch_size = args_dict['batch_size']
        args.learning_rate = args_dict['learning_rate']
        args.att_mode = args_dict['att_mode']
        args.model = args_dict['model']
        args.embed_size = args_dict['embed_size']
        args.hidden_size = args_dict['hidden_size']
        args.num_layers = args_dict['num_layers']
        cur_epoch = checkpoint['epoch']
        print('restore training from existing checkpoint')
        pprint.pprint(args_dict)
    else:
        cur_epoch = 0
        checkpoint = None

    logger.info('building data loader...')
    # Build data loader
    data_loader = get_loader(args.train_image_dir, args.train_caption_path, vocab,
                             transform, args.batch_size,
                             shuffle=True, num_workers=args.num_workers)

    logger.info(f'building model {args.model}...')
    # Build the models
    vocab_size = len(vocab)

    if args.model == 'ssa':
        net = SSA(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size,
                  dropout=args.dropout, fine_tune=args.fine_tune)
    elif args.model == 'nic':
        net = NIC(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size,
                  dropout=args.dropout, fine_tune=args.fine_tune)
    elif args.model == 'scacnn':
        net = SCACNN(embed_dim=args.embed_size, lstm_dim=args.hidden_size, vocab_size=vocab_size,
                     dropout=args.dropout, att_mode=args.att_mode, fine_tune=args.fine_tune)
    else:
        net = None
        print('model name not found: ' + args.model)
        exit(-2)

    net.train()
    net.zero_grad()
    params = net.train_params

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1 and args.model == 'scacnn':
            net = nn.DataParallel(net)
        net.cuda()

    # Loss and Optimizer
    criterion = nn.CrossEntropyLoss(reduce=False)
    optimizer = torch.optim.Adam(params, lr=args.learning_rate)

    if args.restore_train:
        print('restoring from checkpoint...')
        net.load_state_dict(checkpoint['net_state'])
        optimizer.load_state_dict(checkpoint['opt_state'])

    logger.info('start training...')
    # Train the Models
    total_step = len(data_loader)
    running_loss = 0
    for epoch in range(args.num_epochs):
        for i, (images, inputs, targets, masks, lengths, img_ids) in tqdm(enumerate(data_loader), total=total_step,
                                                                          leave=False, ncols=80, unit='b'):
            # Set mini-batch data
            if args.fine_tune:
                images = to_var(images, requires_grad=True)
            else:
                images = to_var(images, requires_grad=False)
            inputs = to_var(inputs, requires_grad=False)
            targets = to_var(targets, requires_grad=False)
            targets = targets.view(-1)
            masks = to_var(masks, requires_grad=False).view(-1)

            net.zero_grad()
            # Forward, Backward and Optimize
            outputs = net.forward(images, inputs, lengths)
            outputs = outputs.contiguous().view(-1, vocab_size)
            loss = criterion(outputs, targets)
            loss = torch.mean(loss * masks)

            loss.backward()
            optimizer.step()
            running_loss += loss.data.item()

            # Make sure python releases GPU memory
            del loss, outputs, images, inputs, targets, masks, lengths, img_ids

        running_loss /= total_step
        logger.info('Epoch [%d/%d], Loss: %.4f, Perplexity: %5.4f'
                        % (cur_epoch + epoch + 1, args.num_epochs, running_loss, np.exp(running_loss)))
        running_loss = 0
        # Save the model
        if (epoch+1) % args.save_step == 0:
            if args.model == 'scacnn':
                save_file = args.model + '-' + args.att_mode + '-model-' + str(cur_epoch + epoch + 1) + '.ckpt'
            else:
                save_file = args.model + '-model-' + str(cur_epoch + epoch+1) + '.ckpt'
            save_path = os.path.join(args.ckpt_dir, save_file)
            args_dict = vars(args)
            opt_state = optimizer.state_dict()
            net_state = net.state_dict()
            epoch_id = epoch + 1
            save_data = {
                'net_state': net_state,
                'opt_state': opt_state,
                'args': args_dict,
                'epoch': epoch_id
            }
            torch.save(save_data, save_path)
            logger.info(f'model saved: {save_path}')


if __name__ == '__main__':
    main()
