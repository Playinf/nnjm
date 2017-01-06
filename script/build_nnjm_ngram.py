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

def get_affiliation(pos, align_vec):
    pos_list = align_vec[pos]
    size = len(align_vec)

    # Affiliation heuristics - see Devlin t al. p1371
    if not pos_list:
        # pos has no alignment, look right, then left, then right-right,
        # then left-left etc
        rpos = pos + 1
        lpos = pos - 1
        while rpos < size or lpos >= 0:
            if rpos < size and align_vec[rpos]:
                pos_list = align_vec[rpos]
                break
            if lpos >= 0 and align_vec[lpos]:
                pos_list = align_vec[lpos]
                break
            rpos += 1
            lpos -= 1

    if not pos_list:
        raise Exception('No alignments in sentence')

    # index of the aligned word in the middle
    midpos = int((len(pos_list) - 1) / 2)
    index = sorted(pos_list)[midpos]

    return index

def get_ngrams(aname, sname, tname, svocab, tvocab, ovocab, m, n, x, oname):
    sfile = open(sname, mode = 'r', encoding = 'utf-8')
    tfile = open(tname, mode = 'r', encoding = 'utf-8')
    afile = open(aname, mode = 'r', encoding = 'utf-8')
    ofile = open(oname, mode = 'w', encoding = 'utf-8')
    fhs = [sfile, tfile, afile]
    mode = ['s2t', 'l2r']

    # get mode
    if x == 1:
        mode[0] = 's2t'
        mode[1] = 'l2r'
    elif x == 2:
        mode[0] = 's2t'
        mode[1] = 'r2l'
    elif x == 3:
        mode[0] = 't2s'
        mode[1] = 'l2r'
    elif x == 4:
        mode[0] = 't2s'
        mode[1] = 'r2l'

    count  = 0
    ngrams = 0
    LOG.info('Extracting ngrams')

    for lines in zip(*fhs):
        stokens = lines[0][:-1].split()
        ttokens = lines[1][:-1].split()
        otokens = (ttokens[:] if mode[0] == 's2t' else stokens[:])
        stokens.insert(0, '<s>')
        ttokens.insert(0, '<s>')
        otokens.insert(0, '<s>')
        stokens.append('</s>')
        ttokens.append('</s>')
        otokens.append('</s>')
        replace_unks(stokens, svocab)
        replace_unks(ttokens, tvocab)
        replace_unks(otokens, ovocab)
        slen = len(stokens)
        tlen = len(ttokens)
        olen = (slen if mode[0] == 't2s' else tlen)
        align_list = []

        if mode[0] == 's2t':
            target_aligns = [[] for t in range(tlen)]
            # BOS alignment
            target_aligns[0] = [0]
        
            for atoken in lines[2][:-1].split():
                spos, tpos = atoken.split('-')
                spos, tpos = int(spos), int(tpos)
                target_aligns[tpos + 1].append(spos + 1)

            # EOS alignment
            target_aligns[-1] = [slen - 1]
            align_list = target_aligns
        else:
            source_aligns = [[] for s in range(slen)]
            # BOS alignment
            source_aligns[0] = [0]
        
            for atoken in lines[2][:-1].split():
                spos, tpos = atoken.split('-')
                spos, tpos = int(spos), int(tpos)
                source_aligns[spos + 1].append(tpos + 1)

            # EOS alignment
            source_aligns[-1] = [tlen - 1]
            align_list = source_aligns

        lctx = 0
        rctx = 0
        ctx = (m if mode[0] == 's2t' else n)
        tokens = (stokens if mode[0] == 's2t' else ttokens)
        token_len = len(tokens)

        if ctx % 2 == 0:
            lctx = int(ctx / 2 - 1)
            rctx = int(ctx / 2)
        else:
            lctx = int(ctx / 2)
            rctx = int(ctx / 2)

        for index in range(olen):
            aind = get_affiliation(index, align_list)
            window = []
            context = []
            outstr = ''

            if mode[1] == 'l2r' and index == 0:
                continue

            if mode[1] == 'r2l' and index == olen - 1:
                continue

            # get affiliation context
            for i in range(lctx):
                pos = aind - lctx + i
                if pos < 0:
                    window.append('<s>')
                else:
                    window.append(tokens[pos])

            window.append(tokens[aind])

            for i in range(rctx):
                pos = aind + i + 1
                if pos >= token_len:
                    window.append('</s>')
                else:
                    window.append(tokens[pos])

            # get n-gram context
            if mode == ['s2t', 'l2r']:
                for i in range(n):
                    pos = index - n + i
                    if pos < 0:
                        context.append('<s>')
                    else:
                        context.append(ttokens[pos])
                outstr = ' '.join(window) + ' ' + ' '.join(context);
                outstr += ' ' + otokens[index]
            elif mode == ['s2t', 'r2l']:
                for i in range(n):
                    pos = index + n - i
                    if pos >= tlen:
                        context.append('</s>')
                    else:
                        context.append(ttokens[pos])
                outstr = ' '.join(window) + ' ' + ' '.join(context);
                outstr += ' ' + otokens[index]
            elif mode == ['t2s', 'l2r']:
                for i in range(m):
                    pos = index - m + i
                    if pos < 0:
                        context.append('<s>')
                    else:
                        context.append(stokens[pos])
                outstr = ' '.join(context) + ' ' + ' '.join(window);
                outstr += ' ' + otokens[index]
            else:
                for i in range(m):
                    pos = index + m - i
                    if pos >= slen:
                        context.append('</s>')
                    else:
                        context.append(stokens[pos])
                outstr = ' '.join(context) + ' ' + ' '.join(window);
                outstr += ' ' + otokens[index]

            ofile.write(outstr + '\n')
            ngrams += 1

        count += 1
        if count % 1000 == 0:
            sys.stderr.write('.')
        if count % 50000 == 0:
            sys.stderr.write(' [%d]\n' % count)

    ofile.close()
    sys.stderr.write('\n')
    LOG.info('Extracted %d ngrams' % ngrams)

def main():
    logging.basicConfig(format = '%(asctime)s %(levelname)s: %(message)s',
                        datefmt = '%Y-%m-%d %H:%M:%S', level = logging.DEBUG)

    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-a', '--source-vocab', type = 'string', dest = 'svocab')
    parser.add_option('-b', '--target-vocab', type = 'string', dest = 'tvocab')
    parser.add_option('-c', '--output-vocab', type = 'string', dest = 'ovocab')
    parser.add_option('-m', '--source-context', type = 'int', dest = 'm')
    parser.add_option('-n', '--target-context', type = 'int', dest = 'n')
    parser.add_option('-o', '--output', type = 'string', dest = 'oname');
    parser.add_option('-r', '--align-file', type = 'string', dest = 'afile')
    parser.add_option('-s', '--source-corpus', type = 'string', dest = 'sfile')
    parser.add_option('-t', '--target-corpus', type = 'string', dest = 'tfile')
    parser.add_option('-w', '--working-dir', type = 'string', dest = 'dir')
    parser.add_option('-v', '--variations', type = 'int', dest = 'model')

    parser.set_defaults(m = 5, n = 4, dir = '.', model = 1)
    opts, args = parser.parse_args(sys.argv)

    if opts.svocab == None or opts.tvocab == None or opts.ovocab == None:
        print('error: not enough vocabulary')
        sys.exit(-1)

    if opts.afile == None or opts.sfile == None or opts.tfile == None:
        print('error: not enough input file')
        sys.exit(-1)

    if opts.oname == None:
        print('error: please input output name')
        sys.exit(-1)

    if opts.model <= 0 or opts.model >= 5:
        print('values for --variation: ')
        print('1: s2t/l2r')
        print('2: s2t/r2l')
        print('3: t2s/l2r')
        print('4: t2s/r2l')
        sys.exit(-1)

    if not os.path.exists(opts.dir):
        os.makedirs(opts.dir)
    else:
        LOG.warn('directory %s already exists, re-using' % opts.dir)

    svocab = set()
    tvocab = set()
    ovocab = set()

    sfile = open(opts.svocab, mode = 'r', encoding = 'utf-8')
    tfile = open(opts.tvocab, mode = 'r', encoding = 'utf-8')
    ofile = open(opts.ovocab, mode = 'r', encoding = 'utf-8')

    for line in sfile:
        line = line.strip()
        word = line.split('|||')[0].strip()
        svocab.add(word)

    for line in tfile:
        line = line.strip()
        word = line.split('|||')[0].strip()
        tvocab.add(word)

    for line in ofile:
        line = line.strip()
        word = line.split('|||')[0].strip()
        ovocab.add(word)

    sfile.close()
    tfile.close()
    ofile.close()
        
    get_ngrams(opts.afile, opts.sfile, opts.tfile, svocab, tvocab, ovocab,
               opts.m, opts.n, opts.model, opts.oname)

if __name__ == '__main__':
    main()
