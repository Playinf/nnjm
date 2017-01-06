#!/usr/bin/python3
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import logging
import optparse
import collections

LOG = logging.getLogger(__name__)

def replace_tags(tokens, tags, vocab):
    for i, t in enumerate(tokens):
        if not t in vocab:
            if i < len(tags):
                tokens[i] = tags[i]
            else:
                print('Error: missing tags for index i:', i)
                print(' '.join(tokens))
                print(' '.join(tags))
                tokens[i] = '<unk>'

# replace tokens not in vocab with <unk>
def replace_unks(tokens, vocab):
    for i, t in enumerate(tokens):
        if not t in vocab:
            tokens[i] = '<unk>'

def get_ngrams(corpus_name, ivocab, ovocab, order, out_name):
    corpus = open(corpus_name, encoding = 'utf-8')
    outfile = open(out_name, mode = 'w', encoding = 'utf-8')

    for line in corpus:
        tokens = line[:-1].split()
        tokens.append('</s>')
        itokens = tokens[:]
        otokens = tokens[:]
        replace_unks(itokens, ivocab)
        replace_unks(otokens, ovocab)

        # context, predicted word
        for pos, tok in enumerate(tokens):
            for i in range(order - 1):
                ind = pos - order + i + 1
                if ind < 0:
                    outfile.write('<s>' + ' ')
                else:
                    outfile.write(itokens[ind] + ' ')
            outfile.write(otokens[pos] + '\n')

    corpus.close()
    outfile.close()

def main():
    logging.basicConfig(format = '%(asctime)s %(levelname)s: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S', level = logging.DEBUG)
    
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-a', '--input-vocab', type = 'string', dest = 'ivocab')
    parser.add_option('-b', '--output-vocab', type = 'string', dest = 'ovocab')
    parser.add_option('-c', '--corpus', type = 'string', dest = 'corpus')
    parser.add_option('-n', '--order', type = 'int', dest = 'n')
    parser.add_option('-o', '--output', type = 'string', dest = 'oname')
    parser.add_option('-w', '--working-dir', type = 'string', dest = 'dir')
    
    
    parser.set_defaults(n = 4, dir = '.')
    opts, args = parser.parse_args(sys.argv)

    if opts.corpus == None or opts.ivocab == None or opts.ovocab == None:
        print('error: must provid input output vocabulary and corpus')
        sys.exit(-1)

    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    else:
        LOG.warn('directory %s already exists, re-using' % opts.dir)

    ivocab = set()
    ovocab = set()

    with open(opts.ivocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            line = line.strip()
            word = line.split('|||')[0].strip()
            ivocab.add(word)

    with open(opts.ovocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            line = line.strip()
            word = line.split('|||')[0].strip()
            ovocab.add(word)
            
    get_ngrams(opts.corpus, ivocab, ovocab, opts.n, opts.oname)

if __name__ == '__main__':
    main()
