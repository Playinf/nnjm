#!/usr/bin/python3
# build vocabulary from file
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import logging
import optparse

import vocab

LOG = logging.getLogger(__name__)

def build_vocab(corpus, inlim, outlim, dir):
    ivocab = vocab.get_pruned_vocab(corpus, inlim)
    ovocab = vocab.get_pruned_vocab(corpus, outlim)

    ivocab.insert(0, '</s>')
    ivocab.insert(0, '<s>')
    ivocab.insert(0, '<null>')
    ivocab.insert(0, '<unk>')

    ovocab.insert(0, '</s>')
    ovocab.insert(0, '<s>')
    ovocab.insert(0, '<unk>')

    vocab.save_vocab(dir, 'input.vocab', ivocab, 0)
    vocab.save_vocab(dir, 'output.vocab', ovocab, 0)
    print('input vocabulary: ', len(ivocab))
    print('output vocabulary: ', len(ovocab))
    
    
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-c', '--corpus', type = 'string', dest = 'corpus')
    parser.add_option('-d', '--working-dir', type = 'string', dest = 'dir')
    parser.add_option('-m', '--input-limit', type = 'int', dest = 'inlim')
    parser.add_option('-n', '--output-limit', type = 'int', dest = 'outlim')

    parser.set_defaults(inlim = 10000, outlim = 10000, dir = '.')
    opts, args = parser.parse_args(sys.argv)

    if opts.corpus == None:
        print('error: no corpus specified')
        sys.exit(-1)

    print('corpus: ', opts.corpus)
    print('working dir: ', opts.dir)
    print('input limit: ', opts.inlim)
    print('output limit: ', opts.outlim)

    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    else:
        LOG.warn('Directory %s already exists, re-using' % opts.dir)

    build_vocab(opts.corpus, opts.inlim, opts.outlim, opts.dir)
