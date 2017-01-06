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

def build_bi_vocab(scorpus, tcorpus, slim, tlim, olim, dir, model):
    mode = ('t2s' if model == 3 or model == 4 else 's2t')
    ocorpus = (tcorpus if mode == 's2t' else scorpus)
    svocab = vocab.get_pruned_vocab(scorpus, slim)
    tvocab = vocab.get_pruned_vocab(tcorpus, tlim)
    ovocab = vocab.get_pruned_vocab(ocorpus, olim)

    svocab.insert(0, '</s>')
    svocab.insert(0, '<s>')
    svocab.insert(0, '<unk>')
    tvocab.insert(0, '</s>')
    tvocab.insert(0, '<s>')
    tvocab.insert(0, '<unk>')
    ovocab.insert(0, '</s>')
    ovocab.insert(0, '<s>')
    ovocab.insert(0, '<unk>')

    if mode == 's2t':
        tvocab.insert(1, '<null>')
    else:
        svocab.insert(1, '<null>')

    vocab.save_vocab(dir, 'source.vocab', svocab, 0)
    vocab.save_vocab(dir, 'target.vocab', tvocab, len(svocab))
    vocab.save_vocab(dir, 'output.vocab', ovocab, 0)
    print('source vocabulary: ', len(svocab))
    print('target vocabulary: ', len(tvocab))
    print('output vocabulary: ', len(ovocab))
    
    
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-s', '--source-corpus', type = 'string', dest = 'sfile')
    parser.add_option('-t', '--target-corpus', type = 'string', dest = 'tfile')
    parser.add_option('-d', '--working-dir', type = 'string', dest = 'dir')
    parser.add_option('-l', '--source-limit', type = 'int', dest = 'slim')
    parser.add_option('-m', '--target-limit', type = 'int', dest = 'tlim')
    parser.add_option('-n', '--output-limit', type = 'int', dest = 'olim')
    parser.add_option('-v', '--variation', type = 'int', dest = 'model')

    parser.set_defaults(dir = '.', slim = 5000, tlim = 5000, olim = 10000,
                        model = 1)
    opts, args = parser.parse_args(sys.argv)

    if opts.sfile == None or opts.tfile == None:
        print('error: not enough input corpus')
        sys.exit(-1)

    if opts.model < 1 or opts.model > 4:
        print('error: wrong value for option -v or --variation')
        print('values allowed for -v or --variation: ')
        print('  1: s2t/l2r nnjm')
        print('  2: s2t/r2l nnjm')
        print('  3: t2s/l2r nnjm')
        print('  4: t2s/r2l nnjm')
        sys.exit(-1)

    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    else:
        LOG.warn('directory %s already exists, re-using' % opts.dir)

    build_bi_vocab(opts.sfile, opts.tfile, opts.slim, opts.tlim, opts.olim,
                   opts.dir, opts.model)
