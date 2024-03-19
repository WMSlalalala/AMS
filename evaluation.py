"""Evaluation"""
from __future__ import print_function
import os
import sys
from data import get_loaders
import torch.nn.functional as F
import time
import numpy as np

import torch
from model import NAAF, xattn_score_test
from collections import OrderedDict
import time
from torch.autograd import Variable
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=0):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / (.0001 + self.count)

    def __str__(self):
        """String representation for logging
        """
        # for values that should be recorded exactly e.g. iteration number
        if self.count == 0:
            return str(self.val)
        # for stats
        return '%.4f (%.4f)' % (self.val, self.avg)
class LogCollector(object):
    """A collection of logging objects that can change from train to val"""

    def __init__(self):
        # to keep the order of logged variables deterministic
        self.meters = OrderedDict()

    def update(self, k, v, n=0):
        # create a new meter if previously not recorded
        if k not in self.meters:
            self.meters[k] = AverageMeter()
        self.meters[k].update(v, n)

    def __str__(self):
        """Concatenate the meters in one log line
        """
        s = ''
        for i, (k, v) in enumerate(self.meters.items()):
            if i > 0:
                s += '  '
            s += k + ' ' + str(v)
        return s

    def tb_log(self, tb_logger, prefix='', step=None):
        """Log using tensorboard
        """
        for k, v in self.meters.items():
            tb_logger.log_value(prefix + k, v.val, step=step)
            
def encode_data(model, data_loader):
    batch_time = AverageMeter()
    val_logger = LogCollector()
    model.val_start()
    end = time.time()
    texts = data_loader.dataset.text
    num_text = len(texts)
    text_bs = 256
    cap_embs1, cap_embs2, cap_embs3,cap_embs4, cap_embs5, cap_embs6  = None, None, None,None, None, None
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = model.tokenizer(text, padding='max_length', truncation=True, max_length=30, return_tensors="pt")
        input_ids = text_input.input_ids.cuda()
        attention_mask = text_input.attention_mask.cuda()
        cap_emb1, cap_emb2, cap_emb3,cap_emb4, cap_emb5, cap_emb6 = model.txt_enc(input_ids, attention_mask)

        # 以下代码处理三个不同的嵌入
        if cap_embs1 is None:
            cap_embs1 = np.zeros((num_text, cap_emb1.size(1), cap_emb1.size(2)))
            cap_embs2 = np.zeros((num_text, cap_emb2.size(1), cap_emb2.size(2)))
            cap_embs3 = np.zeros((num_text, cap_emb3.size(1), cap_emb3.size(2)))
            cap_embs4 = np.zeros((num_text, cap_emb4.size(1), cap_emb4.size(2)))
            cap_embs5 = np.zeros((num_text, cap_emb5.size(1), cap_emb5.size(2)))
            cap_embs6 = np.zeros((num_text, cap_emb6.size(1), cap_emb6.size(2)))
            
        cap_embs1[i: min(num_text, i + text_bs)] = cap_emb1.data.cpu().numpy().copy()
        cap_embs2[i: min(num_text, i + text_bs)] = cap_emb2.data.cpu().numpy().copy()
        cap_embs3[i: min(num_text, i + text_bs)] = cap_emb3.data.cpu().numpy().copy()
        cap_embs4[i: min(num_text, i + text_bs)] = cap_emb4.data.cpu().numpy().copy()
        cap_embs5[i: min(num_text, i + text_bs)] = cap_emb5.data.cpu().numpy().copy()
        cap_embs6[i: min(num_text, i + text_bs)] = cap_emb6.data.cpu().numpy().copy()
        
    img_embs1, img_embs2, img_embs3,img_embs4, img_embs5, img_embs6 = None, None, None,None, None, None
    for i, (images, img_id) in enumerate(data_loader):
        images = images.cuda()
        img_emb1, img_emb2, img_emb3,img_emb4, img_emb5, img_emb6 = model.img_enc(images)

        # 以下代码处理三个不同的嵌入
        if img_embs1 is None:
            img_embs1 = np.zeros((len(data_loader.dataset), img_emb1.size(1), img_emb1.size(2)))
            img_embs2 = np.zeros((len(data_loader.dataset), img_emb2.size(1), img_emb2.size(2)))
            img_embs3 = np.zeros((len(data_loader.dataset), img_emb3.size(1), img_emb3.size(2)))
            img_embs4 = np.zeros((len(data_loader.dataset), img_emb4.size(1), img_emb4.size(2)))
            img_embs5 = np.zeros((len(data_loader.dataset), img_emb5.size(1), img_emb5.size(2)))
            img_embs6 = np.zeros((len(data_loader.dataset), img_emb6.size(1), img_emb6.size(2)))
        # Copy the data to the numpy arrays
        img_embs1[img_id] = img_emb1.data.cpu().numpy().copy()
        img_embs2[img_id] = img_emb2.data.cpu().numpy().copy()
        img_embs3[img_id] = img_emb3.data.cpu().numpy().copy()
        img_embs4[img_id] = img_emb4.data.cpu().numpy().copy()
        img_embs5[img_id] = img_emb5.data.cpu().numpy().copy()
        img_embs6[img_id] = img_emb6.data.cpu().numpy().copy()
    
    return img_embs1, img_embs2, img_embs3,img_embs4, img_embs5, img_embs6, cap_embs1, cap_embs2, cap_embs3,cap_embs4, cap_embs5, cap_embs6

def collate_fn(data):
    data.sort(key=lambda x: len(x[1]), reverse=True)
    images, captions, ids, img_ids, depend, bboxes, Iou, caption_labels, caption_masks = zip(*data)

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    bboxes_ = torch.stack(bboxes, 0)
    Iou = torch.stack(Iou, 0)

    caption_labels_ = torch.stack(caption_labels, 0)
    caption_masks_ = torch.stack(caption_masks, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return images, targets, lengths, ids, depend, bboxes_, Iou, caption_labels_, caption_masks_



def shard_xattn(images, captions, opt, shard_size):
    """
    Computer pairwise t2i image-caption distance with locality sharding
    """
    n_im_shard = (len(images) - 1) // shard_size + 1
    n_cap_shard = (len(captions) - 1) // shard_size + 1
    d1 = np.zeros((len(images), len(captions)))
    for i in range(n_im_shard):
        im_start, im_end = shard_size * i, min(shard_size * (i + 1), len(images))
        for j in range(n_cap_shard):
            sys.stdout.write('\r>> shard_xattn batch (%d,%d)' % (i, j))
            cap_start, cap_end = shard_size * j, min(shard_size * (j + 1), len(captions))
            with torch.no_grad():
                im = torch.from_numpy(images[im_start:im_end]).cuda()
                s = torch.from_numpy(captions[cap_start:cap_end]).cuda()
            similarities = xattn_score_test(im, s, opt)
            d1[im_start:im_end, cap_start:cap_end] = similarities.data.cpu().numpy()
    sys.stdout.write('\n')
    return d1


def i2t(images, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(npts)
    top1 = np.zeros(npts)
    for index in range(npts):
        inds = np.argsort(sims[index])[::-1]
        # Score
        rank = 1e20
        for i in range(5 * index, 5 * index + 5, 1):
            tmp = np.where(inds == i)[0][0]
            if tmp < rank:
                rank = tmp
        ranks[index] = rank
        top1[index] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)


def t2i(images, sims, npts=None, return_ranks=False):
    npts = images.shape[0]
    ranks = np.zeros(5 * npts)
    top1 = np.zeros(5 * npts)
    sims = sims.T

    for index in range(npts):
        for i in range(5):
            inds = np.argsort(sims[5 * index + i])[::-1]
            ranks[5 * index + i] = np.where(inds == index)[0][0]
            top1[5 * index + i] = inds[0]

    # Compute metrics
    r1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    r5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    r10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)
    medr = np.floor(np.median(ranks)) + 1
    meanr = ranks.mean() + 1
    if return_ranks:
        return (r1, r5, r10, medr, meanr), (ranks, top1)
    else:
        return (r1, r5, r10, medr, meanr)
