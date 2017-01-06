#!/usr/bin/python3
# modified: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import logging
import operator

LOG = logging.getLogger(__name__)

def prune_vocab(vocab, prune):
    tup = [(item[0], item[1]) for item in vocab.items()]
    tup = sorted(tup, key = operator.itemgetter(0))
    tup = sorted(tup, key = operator.itemgetter(1), reverse = True)
    vlist = [item[0] for item in tup]

    if prune:
        return vlist[0:prune]
    else:
        return vlist

def get_pruned_vocab(corpus, prune):
    counts = {}
    LOG.info('Reading vocabulary from %s' % corpus)
    lines = 0

    for line in open(corpus, encoding = 'utf-8'):
        for token in line[:-1].split():
            if token not in counts:
                counts[token] = 0
            counts[token] += 1
        lines += 1
        if lines % 1000 == 0:
            sys.stderr.write('.')
        if lines % 50000 == 0:
            sys.stderr.write(' [%d]\n' % lines)

    sys.stderr.write('\n')
    LOG.info('Vocabulary size: %d' % len(counts))

    stokens = ['<s>', '</s>', '<unk>', '<null>']
    for token in stokens:
        if token in counts:
            del counts[token]

    tup = [(item[0], item[1]) for item in counts.items()]
    tup = sorted(tup, key = operator.itemgetter(0))
    tup = sorted(tup, key = operator.itemgetter(1), reverse = True)
    vlist = [item[0] for item in tup]

    if prune:
        return vlist[0:prune]
    else:
        return vlist

def save_vocab(directory, filename, vocab, sid):
    path = os.path.join(directory, filename)
    fh = open(path, mode = 'w', encoding = 'utf-8')
    for word in vocab:
        fh.write(word + ' ||| ' + str(sid) + '\n');
        sid = sid + 1;
