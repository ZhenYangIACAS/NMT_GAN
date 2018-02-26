# -*- coding: utf-8 -*-
#/usr/bin/python2
"""
June 2017 by kyubyong park. 
kbpark.linguist@gmail.com.
https://www.github.com/kyubyong/transformer
Modified by Chunqi Wang in July 2017.
"""
from __future__ import print_function
import codecs
import re as regex
import yaml
from argparse import ArgumentParser
from collections import Counter

from utils import AttrDict


def make_vocab(fpath, fname):
    """Constructs vocabulary.
    
    Args:
      fpath: A string. Input file path.
      fname: A string. Output file name.
    
    Writes vocabulary line by line to `fname`.
    """
    word2cnt = Counter()
    for l in codecs.open(fpath, 'r', 'utf-8'):
        words = l.split()
        word2cnt.update(Counter(words))
    with codecs.open(fname, 'w', 'utf-8') as fout:
        fout.write("{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n{}\t1000000000\n".format("<PAD>", "<UNK>", "<S>", "</S>"))
        for word, cnt in word2cnt.most_common(len(word2cnt)):
            fout.write(u"{}\t{}\n".format(word, cnt))

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    make_vocab(config.train.src_path, config.src_vocab)
    make_vocab(config.train.dst_path, config.dst_vocab)
    print("Done")
