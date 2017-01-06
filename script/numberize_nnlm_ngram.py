#!/usr/bin/python3
# convert ngrams to numerized ngrams
# author: Playinf
# email: playinf@stu.xmu.edu.cn

import os
import sys
import optparse
    
if __name__ == '__main__':
    parser = optparse.OptionParser('%prog [options]')
    parser.add_option('-a', '--input-vocab', type = 'string', dest = 'ivocab')
    parser.add_option('-b', '--output-vocab', type = 'string', dest = 'ovocab')
    parser.add_option('-f', '--file', type = 'string', dest = 'nname')
    parser.add_option('-n', '--order', type = 'int', dest = 'order')
    parser.add_option('-o', '--output', type = 'string', dest = 'oname')

    parser.set_defaults(order = 0)
    opts, args = parser.parse_args(sys.argv)

    if opts.oname == None:
        print('error: no output name specified')
        sys.exit(-1)

    if opts.ivocab == None or opts.ovocab == None:
        print('error: not enough vocabulary')
        sys.exit(-1)

    if opts.nname == None:
        print('error: no input ngram file')
        sys.exit(-1)

    if opts.order == 0:
        print('error: order cannot be 0')
        sys.exit(-1)

    ivocab = {}
    ovocab = {}

    with open(opts.ivocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            vec = line.strip().split('|||')
            name = vec[0].strip()
            id = int(vec[1])
            ivocab[name] = id

    with open(opts.ovocab, mode = 'r', encoding = 'utf-8') as file:
        for line in file:
            vec = line.strip().split('|||')
            name = vec[0].strip()
            id = int(vec[1])
            ovocab[name] = id

    nfile = open(opts.nname, mode = 'r', encoding = 'utf-8')
    ofile = open(opts.oname, mode = 'w', encoding = 'utf-8')
    order = opts.order
    
    for line in nfile:
        vec = line.strip().split(' ')
        vec = [item.strip() for item in vec]
        for i in range(order - 1):
            ofile.write(str(ivocab[vec[i]]) + ' ')
        ofile.write(str(ovocab[vec[order - 1]]))
        ofile.write('\n')

    nfile.close()
    ofile.close()
