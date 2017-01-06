#!/usr/bin/python3
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import logging
import optparse

LOG = logging.getLogger(__name__)

# replace tokens not in vocab with <unk>
def replace_unk(token, vocab):
    if token in vocab:
        return token
    else:
        return '<unk>'

def get_ngrams(sname, tname, aname, oname, ivocab, ovocab, window, model):
    sfile = open(sname, mode = 'r', encoding = 'utf-8')
    tfile = open(tname, mode = 'r', encoding = 'utf-8')
    afile = open(aname, mode = 'r', encoding = 'utf-8')
    ofile = open(oname, mode = 'w', encoding = 'utf-8')
    fhs = [sfile, tfile, afile]
    mode = ('s2t' if model == 1 else 't2s')
    rctx = int(window / 2)
    lctx = window - rctx - 1

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

        for i in range(1, ilen - 1):            
            pos_list = sorted(align_list[i])
            wlist = [otokens[item] for item in pos_list]
            iword = itokens[i]
            oword = ('+++'.join(wlist) if wlist else '<null>')
            iword = replace_unk(iword, ivocab)
            oword = replace_unk(oword, ovocab)
            ngram = []

            for j in range(lctx):
                pos = i - lctx + j
                if (pos < 0):
                    ngram.append('<s>')
                else:
                    ngram.append(replace_unk(itokens[pos], ivocab))

            ngram.append(iword)

            for j in range(rctx):
                pos = i + 1 + j
                if (pos >= ilen):
                    ngram.append('</s>')
                else:
                    ngram.append(replace_unk(itokens[pos], ivocab))

            ngram.append(replace_unk(oword, ovocab))

            ofile.write(' '.join(ngram) + '\n')

    sfile.close()
    tfile.close()
    afile.close()
    ofile.close()

def main():
    logging.basicConfig(format = '%(asctime)s %(levelname)s: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S', level = logging.DEBUG)
    
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-a', '--input-vocab', type = 'string', dest = 'ivocab')
    parser.add_option('-b', '--output-vocab', type = 'string', dest = 'ovocab')
    parser.add_option('-r', '--alignment', type = 'string', dest = 'afile')
    parser.add_option('-s', '--source-corpus', type = 'string', dest = 'sfile')
    parser.add_option('-t', '--target-corpus', type = 'string', dest = 'tfile')
    parser.add_option('-n', '--window', type = 'int', dest = 'n')
    parser.add_option('-o', '--output', type = 'string', dest = 'oname')
    parser.add_option('-w', '--working-dir', type = 'string', dest = 'dir')
    parser.add_option('-v', '--variation', type = 'int', dest = 'model')
    
    parser.set_defaults(n = 14, dir = '.', model = 1)
    opts, args = parser.parse_args(sys.argv)

    if opts.ivocab == None or opts.ovocab == None:
        print('error: must provide input output vocabulary')
        sys.exit(-1)

    if opts.afile == None or opts.sfile == None or opts.tfile == None:
        print('error: must provide input files')
        sys.exit(-1)

    if opts.model < 1 or opts.model > 2:
        print('error: wrong value for option -v')
        print('values for option -v or --variation:')
        print('  1: s2t nnltm')
        print('  2: t2s nnltm')
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
            
    get_ngrams(opts.sfile, opts.tfile, opts.afile, opts.oname, ivocab, ovocab,
               opts.n, opts.model)

if __name__ == '__main__':
    main()
