#!/usr/bin/python3
# convert ngrams to numerized ngrams
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import optparse
    
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-a', '--source-vocab', type = 'string', dest = 'svocab')
    parser.add_option('-b', '--target-vocab', type = 'string', dest = 'tvocab')
    parser.add_option('-c', '--output-vocab', type = 'string', dest = 'ovocab')
    parser.add_option('-f', '--file', type = 'string', dest = 'nname')
    parser.add_option('-m', '--source-context', type = 'int', dest = 'sctx')
    parser.add_option('-n', '--target-context', type = 'int', dest = 'tctx')
    parser.add_option('-o', '--output', type = 'string', dest = 'oname')

    parser.set_defaults(snum = 0, tnum = 0)
    opts, args = parser.parse_args(sys.argv)

    if opts.oname == None:
        print('error: no output name specified')
        sys.exit(-1)

    if opts.svocab == None or opts.tvocab == None or opts.ovocab == None:
        print('error: not enough vocabulary')
        sys.exit(-1)

    if opts.nname == None:
        print('error: no input ngram file')
        sys.exit(-1)

    if opts.sctx == 0 or opts.tctx == 0:
        print('error: context size cannot be 0')
        sys.exit(-1)

    svocab = {}
    tvocab = {}
    ovocab = {}

    with open(opts.svocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            vec = line.strip().split('|||')
            name = vec[0].strip()
            id = int(vec[1])
            svocab[name] = id

    with open(opts.tvocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            vec = line.strip().split('|||')
            name = vec[0].strip()
            id = int(vec[1])
            tvocab[name] = id

    with open(opts.ovocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            vec = line.strip().split('|||')
            name = vec[0].strip()
            id = int(vec[1])
            ovocab[name] = id

    nfile = open(opts.nname, mode = 'r', encoding = 'utf-8')
    ofile = open(opts.oname, mode = 'w', encoding = 'utf-8')
    snum = opts.sctx;
    tnum = opts.tctx
    
    for line in nfile:
        vec = line.strip().split(' ')
        vec = [item.strip() for item in vec]
        slist = vec[0:snum]
        tlist = vec[snum:snum + tnum]
        for item in slist:
            ofile.write(str(svocab[item]) + ' ')
        for item in tlist:
            ofile.write(str(tvocab[item]) + ' ')
        ofile.write(str(ovocab[vec[snum + tnum]]))
        ofile.write('\n')

    nfile.close()
    ofile.close()
