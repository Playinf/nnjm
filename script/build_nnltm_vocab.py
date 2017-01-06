#!/usr/bin/python3
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import logging
import optparse

import vocab

LOG = logging.getLogger(__name__)

def build_nnltm_vocab(scorpus, tcorpus, acorpus, ilim, olim, dir, model):
    sfile = open(scorpus, mode = 'r', encoding = 'utf-8')
    tfile = open(tcorpus, mode = 'r', encoding = 'utf-8')
    afile = open(acorpus, mode = 'r', encoding = 'utf-8')
    fhs = [sfile, tfile, afile]
    mode = ('s2t' if model == 1 else 't2s')
    icounts = {}
    ocounts = {}
    
    for lines in zip(*fhs):
        stokens = lines[0][:-1].split()
        ttokens = lines[1][:-1].split()
        itokens = (stokens if mode == 's2t' else ttokens)
        otokens = (ttokens if mode == 's2t' else stokens)
        itokens.insert(0, '<s>')
        otokens.insert(0, '<s>')
        itokens.append('</s>')
        otokens.append('</s>')
        slen = len(stokens)
        tlen = len(ttokens)
        ilen = len(itokens)
        olen = len(otokens)
        align_list = []

        if mode == 't2s':
            target_aligns = [[] for t in range(tlen)]
            target_aligns[0] = [0]
        
            for atoken in lines[2][:-1].split():
                spos, tpos = atoken.split('-')
                spos, tpos = int(spos), int(tpos)
                target_aligns[tpos + 1].append(spos + 1)

            target_aligns[-1] = [slen - 1]
            align_list = target_aligns
        else:
            source_aligns = [[] for s in range(slen)]
            source_aligns[0] = [0]
        
            for atoken in lines[2][:-1].split():
                spos, tpos = atoken.split('-')
                spos, tpos = int(spos), int(tpos)
                source_aligns[spos + 1].append(tpos + 1)

            source_aligns[-1] = [tlen - 1]
            align_list = source_aligns

        for i in range(ilen):
            pos_list = sorted(align_list[i])
            iword = itokens[i]

            # add source vocab
            if iword not in icounts:
                icounts[iword] = 1
            else:
                icounts[iword] += 1

            # add target vocab
            if pos_list:
                wlist = [otokens[item] for item in pos_list]
                oword = '+++'.join(wlist)
                if oword not in ocounts:
                    ocounts[oword] = 1
                else:
                    ocounts[oword] += 1

    sp_tokens = ['<s>', '</s>', '<unk>', '<null>']

    for token in sp_tokens:
        if token in icounts:
            del icounts[token]

    for token in sp_tokens:
        if token in ocounts:
            del ocounts[token]

    ivocab = vocab.prune_vocab(icounts, ilim)
    ovocab = vocab.prune_vocab(ocounts, olim)

    ivocab.insert(0, '</s>')
    ivocab.insert(0, '<s>')
    ivocab.insert(0, '<null>')
    ivocab.insert(0, '<unk>')
    ovocab.insert(0, '</s>')
    ovocab.insert(0, '<s>')
    ovocab.insert(0, '<null>')
    ovocab.insert(0, '<unk>')

    vocab.save_vocab(dir, 'input.vocab', ivocab, 0)
    vocab.save_vocab(dir, 'output.vocab', ovocab, 0)
    print('input vocabulary: ', len(ivocab))
    print('output vocabulary: ', len(ovocab))

    sfile.close()
    tfile.close()
    afile.close()
    
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-r', '--alignment', type = 'string', dest = 'afile')
    parser.add_option('-s', '--source-corpus', type = 'string', dest = 'sfile')
    parser.add_option('-t', '--target-corpus', type = 'string', dest = 'tfile')
    parser.add_option('-d', '--working-dir', type = 'string', dest = 'dir')
    parser.add_option('-m', '--input-limit', type = 'int', dest = 'ilim')
    parser.add_option('-n', '--output-limit', type = 'int', dest = 'olim')
    parser.add_option('-v', '--variation', type = 'int', dest = 'model')

    parser.set_defaults(dir = '.', ilim = 5000, olim = 10000, model = 1)
    opts, args = parser.parse_args(sys.argv)

    if opts.sfile == None or opts.tfile == None or opts.afile == None:
        print('error: not enough input corpus')
        sys.exit(-1)

    if opts.model < 1 or opts.model > 2:
        print('error: wrong value for option -v or --variation')
        print('values allowed for -v or --variation: ')
        print('  1: s2t nnltm')
        print('  2: t2s nnltm')
        sys.exit(-1)

    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    else:
        LOG.warn('directory %s already exists, re-using' % opts.dir)

    build_nnltm_vocab(opts.sfile, opts.tfile, opts.afile, opts.ilim, opts.olim,
                      opts.dir, opts.model)
