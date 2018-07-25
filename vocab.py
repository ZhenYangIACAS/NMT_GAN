# -*- coding: utf-8 -*-
#/usr/bin/python2

import numpy
import cPickle as pkl
#import ipdb
import sys
import fileinput

from collections import OrderedDict

def main():
    for filename in sys.argv[1:]:
        print 'Processing', filename
        word_freqs = OrderedDict()
        with open(filename, 'r') as f:
            for line in f:
                words_in = line.strip().split(' ')
                for w in words_in:
                    if w not in word_freqs:
                        word_freqs[w] = 0
                    word_freqs[w] += 1
        words = word_freqs.keys()
        freqs = word_freqs.values()
        #print freqs
        #ipdb.set_trace()
        sorted_idx = numpy.argsort(freqs)
        #print sorted_idx
        sorted_words = [words[ii] for ii in sorted_idx[::-1]]

        worddict = OrderedDict()
        worddict['<PAD>'] = 0
        worddict['<UNK>'] = 1
        worddict['<S>'] = 1
        worddict['</S>'] = 1
        for ii, ww in enumerate(sorted_words):
            worddict[ww] = ii+4

        with open('%s.voc'%filename, 'wb') as f:
            for key, _ in worddict.items():
                f.write(key +'\n')

        print 'Done'

if __name__ == '__main__':
    main()
