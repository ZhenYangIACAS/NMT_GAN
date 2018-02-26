from __future__ import print_function
import numpy as np
import os
import codecs
import logging
from tempfile import mkstemp
from itertools import izip


class AttrDict(dict):
    """
    Dictionary whose keys can be accessed as attributes.
    """

    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)

    def __getattr__(self, item):
        if type(self[item]) is dict:
            self[item] = AttrDict(self[item])
        return self[item]


class DataUtil(object):
    """
    Util class for creating batches for training and testing.
    """
    def __init__(self, config):
        self.config = config
        self._logger = logging.getLogger('util')
        #self.load_vocab()

    def load_vocab(self,
                   src_vocab=None,
                   dst_vocab=None,
                   src_vocab_size=None,
                   dst_vocab_size=None):
        """
        Load vocab from disk. The fisrt four items in the vocab should be <PAD>, <UNK>, <S>, </S>
        """

        def load_vocab_(path, vocab_size):
            vocab = [line.split()[0] for line in codecs.open(path, 'r', 'utf-8')]
            vocab = vocab[:vocab_size]
            assert len(vocab) == vocab_size
            word2idx = {word: idx for idx, word in enumerate(vocab)}
            idx2word = {idx: word for idx, word in enumerate(vocab)}
            return word2idx, idx2word

        if src_vocab and dst_vocab and src_vocab_size and dst_vocab_size:
            self._logger.debug('Load set vocabularies as %s and %s.' % (src_vocab, dst_vocab))
            self.src2idx, self.idx2src = load_vocab_(src_vocab, src_vocab_size)
            self.dst2idx, self.idx2dst = load_vocab_(dst_vocab, dst_vocab_size)
        else:
            self._logger.debug('Load vocabularies %s and %s.' % (self.config.src_vocab, self.config.dst_vocab))
            self.src2idx, self.idx2src = load_vocab_(self.config.src_vocab, self.config.src_vocab_size)
            self.dst2idx, self.idx2dst = load_vocab_(self.config.dst_vocab, self.config.dst_vocab_size)

    def get_training_batches(self,
                             shuffle=True,
                             set_train_src_path=None,
                             set_train_dst_path=None,
                             set_batch_size=None,
                             set_max_length=None):
        """
        Generate batches with fixed batch size.
        """
        if set_train_src_path and set_train_dst_path:
            src_path=set_train_src_path
            dst_path=set_train_dst_path
        else:
            src_path = self.config.train.src_path
            dst_path = self.config.train.dst_path

        if set_batch_size:
            batch_size=set_batch_size
        else:
            batch_size = self.config.train.batch_size

        if set_max_length:
            max_length=set_max_length
        else:
            max_length = self.config.train.max_length

        # Shuffle the training files.
        if shuffle:
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        src_sents, dst_sents = [], []
        for src_sent, dst_sent in izip(codecs.open(src_shuf_path, 'r', 'utf8'),
                                       codecs.open(dst_shuf_path, 'r', 'utf8')):
            # If exceed the max length, abandon this sentence pair.
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            if  len(src_sent) > max_length or len(dst_sent) > max_length:
                continue
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents = [], []

        if src_sents and dst_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    def get_training_batches_with_buckets(self, shuffle=True):
        """
        Generate batches according to bucket setting.
        """

        buckets = [(i, i) for i in range(10, 100, 5)] + [(self.config.train.max_length, self.config.train.max_length)]

        def select_bucket(sl, dl):
            for l1, l2 in buckets:
                if sl < l1 and dl < l2:
                    return (l1, l2)
            return None

        # Shuffle the training files.
        src_path = self.config.train.src_path
        dst_path = self.config.train.dst_path
        if shuffle:
            self._logger.debug('Shuffle files %s and %s.' % (src_path, dst_path))
            src_shuf_path, dst_shuf_path = self.shuffle([src_path, dst_path])
        else:
            src_shuf_path = src_path
            dst_shuf_path = dst_path

        caches = {}
        for bucket in buckets:
            caches[bucket] = [[], [], 0, 0]  # src sentences, dst sentences, src tokens, dst tokens

        for src_sent, dst_sent in izip(codecs.open(src_shuf_path, 'r', 'utf8'),
                                       codecs.open(dst_shuf_path, 'r', 'utf8')):
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()

            bucket = select_bucket(len(src_sent), len(dst_sent))
            if bucket is None:  # No bucket is selected when the sentence length exceed the max length.
                continue

            caches[bucket][0].append(src_sent)
            caches[bucket][1].append(dst_sent)
            caches[bucket][2] += len(src_sent)
            caches[bucket][3] += len(dst_sent)

            if max(caches[bucket][2], caches[bucket][3]) >= self.config.train.tokens_per_batch:
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
                self._logger.debug(
                    'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
                yield batch
                caches[bucket] = [[], [], 0, 0]

        # Clean remain sentences.
        for bucket in buckets:
            # Ensure each device at least get one sample.
            if len(caches[bucket][0]) > len(self.config.train.devices.split(',')):
                batch = (self.create_batch(caches[bucket][0], o='src'), self.create_batch(caches[bucket][1], o='dst'))
            self._logger.debug(
                'Yield batch with source shape %s and target shape %s.' % (batch[0].shape, batch[1].shape))
            yield batch

        # Remove shuffled files when epoch finished.
        if shuffle:
            os.remove(src_shuf_path)
            os.remove(dst_shuf_path)

    @staticmethod
    def shuffle(list_of_files):
        tf_os, tpath = mkstemp()
        tf = open(tpath, 'w')

        fds = [open(ff) for ff in list_of_files]

        for l in fds[0]:
            lines = [l.strip()] + [ff.readline().strip() for ff in fds[1:]]
            print("|||||".join(lines), file=tf)

        [ff.close() for ff in fds]
        tf.close()

        os.system('shuf %s > %s' % (tpath, tpath + '.shuf'))

        fds = [open(ff + '.{}.shuf'.format(os.getpid()), 'w') for ff in list_of_files]

        for l in open(tpath + '.shuf'):
            s = l.strip().split('|||||')
            for i, fd in enumerate(fds):
                print(s[i], file=fd)

        [ff.close() for ff in fds]

        os.remove(tpath)
        os.remove(tpath + '.shuf')

        return [ff + '.{}.shuf'.format(os.getpid()) for ff in list_of_files]

    def get_test_batches(self,
                         set_src_path=None,
                         set_batch=None):
        if set_src_path and set_batch:
            src_path=set_src_path
            batch_size=set_batch
        else:
            src_path = self.config.test.src_path
            batch_size = self.config.test.batch_size

        # Read batches from test files.
        src_sents = []
        for src_sent in codecs.open(src_path, 'r', 'utf8'):
            src_sent = src_sent.split()
            src_sents.append(src_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src')
                src_sents = []
        if src_sents:
            yield self.create_batch(src_sents, o='src')

    def get_test_batches_with_target(self,
                                     set_test_src_path=None,
                                     set_test_dst_path=None,
                                     set_batch_size=None):
        """
        Usually we don't need target sentences for test unless we want to compute PPl.
        Returns:
            Paired source and target batches.
        """
        if set_test_src_path and set_test_dst_path and set_batch_size:
            src_path=set_test_src_path
            dst_path=set_test_dst_path
            batch_size=set_batch_size

        else:
            src_path = self.config.test.src_path
            dst_path = self.config.test.dst_path
            batch_size = self.config.test.batch_size

        # Read batches from test files.
        src_sents, dst_sents = [], []
        for src_sent, dst_sent in izip(codecs.open(src_path, 'r', 'utf8'),
                                       codecs.open(dst_path, 'r', 'utf8')):
            src_sent = src_sent.split()
            dst_sent = dst_sent.split()
            src_sents.append(src_sent)
            dst_sents.append(dst_sent)
            # Create a padded batch.
            if len(src_sents) >= batch_size:
                yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')
                src_sents, dst_sents = [], []
        if src_sents:
            yield self.create_batch(src_sents, o='src'), self.create_batch(dst_sents, o='dst')

    def create_batch(self, sents, o):
        # Convert words to indices.
        assert o in ('src', 'dst')
        word2idx = self.src2idx if o == 'src' else self.dst2idx
        indices = []
        for sent in sents:
            x = [word2idx.get(word, 1) for word in (sent + [u"</S>"])]  # 1: OOV, </S>: End of Text
            indices.append(x)

        # Pad to the same length.
        maxlen = max([len(s) for s in indices])
        X = np.zeros([len(indices), maxlen], np.int32)
        for i, x in enumerate(indices):
            X[i, :len(x)] = x

        return X

    def indices_to_words(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        sents = []
        for y in Y: # for each sentence
            sent = []
            for i in y: # For each word
                if i == 3:  # </S>
                    break
                w = idx2word[i]
                sent.append(w)
            sents.append(' '.join(sent))
        return sents

    def indices_to_words_del_pad(self, Y, o='dst'):
        assert o in ('src', 'dst')
        idx2word = self.idx2src if o == 'src' else self.idx2dst
        pad_index = idx2word
        sents=[]
        for y in Y:
            sent= []
            for i in y:
                if i > 0:
                    w = idx2word[i]
                    sent.append(w)
            sents.append(' '.join(sent))
        return sents



