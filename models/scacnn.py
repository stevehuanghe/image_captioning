"""
The SCA-CNN model
"""

import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable


class SCACNN(nn.Module):
    def __init__(self, vocab_size, embed_dim=100, lstm_dim=1000, k=512, dropout=0, fine_tune=False, att_mode=None):
        super(SCACNN, self).__init__()

        self.train_params = []
        self.att_mode = att_mode

        """Load the pretrained ResNet-152 and replace top fc layer."""
        resnet = models.resnet152(pretrained=True)
        modules = list(resnet.children())[:-1]  # delete the last fc layer.
        block5 = list(modules[-2].children())  # get the last block
        bottle = list(block5[-1].children())  # get the last residual unit
        block5 = block5[:-1] + bottle[:-2]  # res5c_branch2c conv layer
        modules = modules[:-2] + block5  # add the block back to resnet

        self.resnet = nn.Sequential(*modules)  # output shape=[batch, 2048, 7, 7]
        self.resnet_params = list(resnet.parameters())
        if fine_tune:
            self.train_params += self.resnet_params
        else:
            for param in self.resnet_params:
                param.requires_grad = False

        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.lstm_dim = lstm_dim
        self.k = k

        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.train_params += list(self.embed.parameters())

        self.lstm = nn.LSTM(self.embed_dim + 2048*49, lstm_dim, 1)
        self.dropout = nn.Dropout(p=dropout)
        self.train_params += list(self.lstm.parameters())

        self.softmax = nn.Softmax(dim=1)
        self.tanh = nn.Tanh()

        self.s_linear_v = nn.Linear(2048, k)  # out=[batch, 49, k]
        self.train_params += list(self.s_linear_v.parameters())
        self.s_linear_h = nn.Linear(lstm_dim, k, bias=False)  # out=[batch, k]
        self.train_params += list(self.s_linear_h.parameters())
        self.s_linear_a = nn.Linear(k, 1)  # out=[batch, 49, 1]
        self.train_params += list(self.s_linear_a.parameters())

        self.c_linear_v = nn.Linear(1, self.k)  # out=[batch, 2048, k]
        self.train_params += list(self.c_linear_v.parameters())
        self.c_linear_h = nn.Linear(lstm_dim, k, bias=False)  # out=[batch, k]
        self.train_params += list(self.c_linear_h.parameters())
        self.c_linear_b = nn.Linear(k, 1)  # out=[batch, 2048, 1]
        self.train_params += list(self.c_linear_b.parameters())

        self.lstm_output = nn.Linear(self.lstm_dim, self.vocab_size)
        self.train_params += list(self.lstm_output.parameters())

    def spatial_attention(self, V, h):
        """
        calculate the spatial attention weights
        :param V: shape=[batch, 49, 2048]
        :param h: shape=[batch, d]
        :return alpha: shape=[batch, 49, 1]
        """
        v_feats = self.s_linear_v(V)
        h_feats = self.s_linear_h(h).view(-1, 1, self.k)  # [batch, 49, k]
        a = self.tanh(v_feats + h_feats)  # [batch, 49, k]
        alpha = self.softmax(self.s_linear_a(a))  # [batch, 49, 1]
        return alpha

    def channel_attention(self, V, h):
        """
        calculate the channel-wise attention weights
        :param V: shape=[batch, 49, 2048]
        :param h: shape=[batch, d]
        :return beta: shape=[batch, 2048, 1]
        """
        V1 = V.permute(0, 2, 1)  # [batch, 2048, 49]
        v = torch.mean(V1, 2, keepdim=True)  # [batch, 2048, 1]
        v_feats = self.c_linear_v(v)  # [batch, 2048, k]
        h_feats = self.c_linear_h(h).view(-1, 1, self.k)  # [batch, 1, k]
        b = self.tanh(v_feats + h_feats)  # [batch, 2048, k]
        beta = self.softmax(self.c_linear_b(b))  # [batch, 2048, 1]
        return beta.permute(0, 2, 1)  # [batch, 1, 2048]

    def forward(self, images, captions, lengths):
        """Extract the image feature vectors."""
        batch_size = images.size()[0]
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(batch_size, 2048, 49).permute(0, 2, 1)  # [batch, 49, 2048]

        embeddings = self.embed(captions).permute(1, 0, 2)  # [length, batch, embed_dim]
        embeddings = torch.split(embeddings, 1)

        logits = []
        seq_max_len = max(lengths)
        hidden = Variable(torch.zeros(batch_size, self.lstm_dim), requires_grad=False)
        states = None

        if torch.cuda.is_available():
            hidden = hidden.cuda()

        for t in range(seq_max_len):
            if self.att_mode == 'cs':
                beta = self.channel_attention(features, hidden)
                features = beta * features
                alpha = self.spatial_attention(features, hidden)
                feats = alpha * features
            elif self.att_mode == 'c':
                beta = self.channel_attention(features, hidden)
                feats = beta * features
            elif self.att_mode == 's':
                alpha = self.spatial_attention(features, hidden)
                feats = alpha * features
            else:
                alpha = self.spatial_attention(features, hidden)
                features = alpha * features
                beta = self.channel_attention(features, hidden)
                feats = beta * features

            feats = feats.view(1, batch_size, -1)
            embed = embeddings[t]
            inputs = torch.cat([embed, feats], dim=2)
            hidden, states = self.lstm(inputs, states)
            hidden = self.dropout(hidden)
            output = self.lstm_output(hidden)
            logits.append(output)

        logits = torch.cat(logits, dim=0).permute(1, 0, 2)  # [batch, length, vocab_size]
        return logits

    def greedy_search(self, images, start_token: int):
        batch_size = images.size()[0]
        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(batch_size, 2048, 49).permute(0, 2, 1)  # [batch, 49, 2048]

        captions = []
        seq_max_len = 20
        hidden = Variable(torch.zeros(batch_size, self.lstm_dim), requires_grad=False)
        states = None
        word = Variable(torch.Tensor([start_token]).repeat(batch_size, 0).long(), requires_grad=False)
        if torch.cuda.is_available():
            hidden = hidden.cuda()
            word = word.cuda()

        for t in range(seq_max_len):
            if self.att_mode == 'cs':
                beta = self.channel_attention(features, hidden)
                features = beta * features
                alpha = self.spatial_attention(features, hidden)
                feats = alpha * features
            elif self.att_mode == 'c':
                beta = self.channel_attention(features, hidden)
                feats = beta * features
            elif self.att_mode == 's':
                alpha = self.spatial_attention(features, hidden)
                feats = alpha * features
            else:
                alpha = self.spatial_attention(features, hidden)
                features = alpha * features
                beta = self.channel_attention(features, hidden)
                feats = beta * features
            feats = feats.view(1, batch_size, -1)
            embed = self.embed(word).view(1, batch_size, -1)
            inputs = torch.cat([embed, feats], dim=2)
            hidden, states = self.lstm(inputs, states)
            output = self.lstm_output(hidden)
            predicted = output.max(2)[1]
            captions.append(predicted)
            word = predicted

        captions = torch.cat(captions, dim=0).permute(1, 0)  # [batch, length]
        return captions

    def beam_search(self, images, start_token: int, seq_max_len=20, beam_width=5):
        batch_size = images.size()[0]

        features = self.resnet(images)
        features = Variable(features.data)
        features = features.view(features.size(0), -1)
        features = self.bn(self.linear(features))

        feats = features.view(1, batch_size, -1)

        hidden, states = self.lstm(feats)
        words = Variable(torch.Tensor([start_token]).long(), requires_grad=False).repeat(batch_size).view(batch_size, 1,
                                                                                                          1)
        probs = Variable(torch.zeros(batch_size, 1))  # [batch, beam]
        if torch.cuda.is_available():
            words = words.cuda()
            probs = probs.cuda()
        h, c = states

        all_hidden = h.unsqueeze(3)  # [1, batch, lstm_dim, beam]
        all_cell = c.unsqueeze(3)  # [1, batch, lstm_dim, beam]
        all_words = words  # [batch, length, beam]
        all_probs = probs  # [batch, beam]
        for t in range(seq_max_len):
            new_words = []
            new_cell = []
            new_hidden = []
            new_probs = []
            tmp_words = all_words.split(1, 2)
            tmp_probs = all_probs.split(1, 1)
            tmp_hidden = all_hidden.split(1, 3)
            tmp_cell = all_cell.split(1, 3)
            for i in range(len(tmp_words)):
                last_word = tmp_words[i].split(1, 1)[-1].view(batch_size)
                if self.att_mode == 'cs':
                    beta = self.channel_attention(features, hidden)
                    features = beta * features
                    alpha = self.spatial_attention(features, hidden)
                    feats = alpha * features
                elif self.att_mode == 'c':
                    beta = self.channel_attention(features, hidden)
                    feats = beta * features
                elif self.att_mode == 's':
                    alpha = self.spatial_attention(features, hidden)
                    feats = alpha * features
                else:
                    alpha = self.spatial_attention(features, hidden)
                    features = alpha * features
                    beta = self.channel_attention(features, hidden)
                    feats = beta * features
                feats = feats.view(1, batch_size, -1)
                embed = self.embed(last_word).view(1, batch_size, -1).contiguous()
                inputs = torch.cat([embed, feats], dim=2)

                last_state = (tmp_hidden[i].squeeze(3).contiguous(), tmp_cell[i].squeeze(3).contiguous())
                hidden, states = self.lstm(inputs, last_state)

                probs = self.log_softmax(self.lstm_output(hidden))  # [1, batch, vocab_size]

                probs, indices = probs.topk(beam_width, 2)
                probs = probs.view(batch_size, beam_width)  # [batch, beam]
                indices = indices.permute(1, 0, 2)  # [batch, 1, beam]

                tmp_words_rep = tmp_words[i].repeat(1, 1, beam_width)

                probs_cand = tmp_probs[i] + probs  # [batch, beam]
                words_cand = torch.cat([tmp_words_rep, indices], 1)  # [batch, length+1, beam]
                hidden_cand = states[0].unsqueeze(3).repeat(1, 1, 1, beam_width)  # [1, batch, lstm_dim, beam]
                cell_cand = states[1].unsqueeze(3).repeat(1, 1, 1, beam_width)

                new_words.append(words_cand)
                new_probs.append(probs_cand)
                new_hidden.append(hidden_cand)
                new_cell.append(cell_cand)

            new_words = torch.cat(new_words, 2)  # [batch, length+1, beam*beam]
            new_probs = torch.cat(new_probs, 1)  # [batch, beam*beam]
            new_cell = torch.cat(new_cell, 3)  # [1, batch, lstm_dim, beam*beam]
            new_hidden = torch.cat(new_hidden, 3)  # [1, batch, lstm_dim, beam*beam]

            probs, idx = new_probs.topk(beam_width, 1)  # [batch, beam]
            idx_words = idx.view(batch_size, 1, beam_width)
            idx_words = idx_words.repeat(1, t + 2, 1)
            idx_states = idx.view(1, batch_size, 1, beam_width).repeat(1, 1, self.lstm_dim, 1)

            # reduce the beam*beam candidates to top@beam candidates
            all_probs = probs
            all_words = new_words.gather(2, idx_words)
            all_hidden = new_hidden.gather(3, idx_states)
            all_cell = new_cell.gather(3, idx_states)

        idx = all_probs.argmax(1)  # [batch]
        idx = idx.view(batch_size, 1, 1).repeat(1, seq_max_len + 1, 1)
        captions = all_words.gather(2, idx).squeeze(2)  # [batch, length]

        return captions


