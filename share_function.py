from __future__ import division

import tensorflow as tf
import numpy
import time
import os
from operator import mul

from data_iterator import disTextIterator
from data_iterator import genTextIterator
from data_iterator import TextIterator

from collections import defaultdict
from math import exp


def prepare_gan_dis_data(train_data_source, train_data_target, gan_dis_source_data, gan_dis_positive_data,
                         num=None, reshuf=True):

    source = open(train_data_source, 'r')
    sourceLists = source.readlines()

    if num is None or num > len(sourceLists):
            num = len(sourceLists)
   
    if reshuf:
            os.popen('python shuffle.py ' +train_data_source+' '+train_data_target)
            os.popen('head -n ' + str(num) +' '+ train_data_source+'.shuf'+' >'+gan_dis_source_data)
            os.popen('head -n ' + str(num) +' '+ train_data_target+'.shuf'+' >'+gan_dis_positive_data)
    else:
            os.popen('head -n ' + str(num) +' '+ train_data_source + '.shuf' + ' >'+gan_dis_source_data)
            os.popen('head -n ' + str(num) +' '+ train_data_target + '.shuf' + ' >'+gan_dis_positive_data)
    
    os.popen('rm '+train_data_source+'.shuf')
    os.popen('rm '+train_data_target+'.shuf')
    return num

def prepare_three_gan_dis_dev_data(gan_dis_positive_data, gan_dis_negative_data, gan_dis_source_data, dev_dis_positive_data, dev_dis_negative_data, dev_dis_source_data, num):
    gan_dis = open(gan_dis_positive_data, 'r')
    disLists = gan_dis.readlines()
  
    if num is None or num > len(disLists):
            num = len(disLists)

    os.popen('head -n '+ str(num) +' '+gan_dis_positive_data+' >'+dev_dis_positive_data)
    os.popen('head -n '+ str(num) +' '+gan_dis_negative_data+' >'+dev_dis_negative_data)
    os.popen('head -n '+ str(num) +' '+gan_dis_source_data+' >'+dev_dis_source_data)

    return num
 
def prepare_gan_dis_dev_data(gan_dis_positive_data, gan_dis_negative_data, dev_dis_positive_data, dev_dis_negative_data, num):

    gan_dis = open(gan_dis_positive_data, 'r')
    disLists = gan_dis.readlines()

    if num is None or num > len(disLists):
            num = len(disLists)
            
    os.popen('head -n '+ str(num) +' '+gan_dis_positive_data+' >'+dev_dis_positive_data)
    os.popen('head -n '+ str(num) +' '+gan_dis_negative_data+' >'+dev_dis_negative_data)

    return num

def print_string(src_or_trg, indexs, worddicts_r):
    sample_str = ''
    for index in indexs:
        if index > 0:
           if src_or_trg == 'y':
               word_str = worddicts_r[1][index]
           else:
               word_str = worddicts_r[0][index]
           sample_str = sample_str + word_str + ' '
    return sample_str

class FlushFile:
    """
    A wrapper for File, allowing users see result immediately.
    """
    def __init__(self, f):
        self.f = f

    def write(self, x):
        self.f.write(x)
        self.f.flush()

def _p(pp, name):
        return '%s_%s' % (pp, name)

def dis_train_iter(dis_positive_data, dis_negative_data, reshuffle, dictionary, n_words_trg, batch_size, maxlen):
    iter = 0
    while True:
       if reshuffle:
          os.popen('python shuffle.py '+dis_positive_data+' '+dis_positive_data)
          os.popen('mv ' + dis_negative_data + '.shuf ' + dis_negtive_data)
          os.popen('mv ' + dis_negative_data + '.shuf ' + dis_negative_data)
       disTrain = disTextIterator(dis_positive_data, dis_negative_data, dictionary, batch_size, maxlen, n_words_trg)
       iter +=1
       ExampleNum = 0
       iterStart = time.time()
       for x, y in disTrain:
          ExampleNum += len(x)
          yield x, y, iter
       TimeCost = time.time() - EpochStart
    print('Seen ', ExampleNum, ' examples for discriminator. Time cost : ', TimeCost)


def gen_train_iter(gen_file, reshuffle, dictionary, n_words, batch_size, maxlen):
   iter = 0
   while True:
      if reshuffle:
          os.popen('python shuffle.py '+ gen_file)
          os.popen('mv '+ gen_file +'.shuf ' + gen_file)
      gen_train = genTextIterator(gen_file, dictionary, n_words_source = n_words, batch_size = batch_size, maxlen=maxlen)
      ExampleNum = 0
      EpochStart = time.time()
      for x in gen_train:
          if len(x) < batch_size:
                  continue
          ExampleNum +=len(x)
          yield x, iter
      TimeCost = time.time() - EpochStart
      iter +=1
      print('Seen ', ExampleNum, 'generator samples. Time cost is ', TimeCost)

def gen_force_train_iter(source_data, target_data, reshuffle, source_dict, target_dict, batch_size, maxlen, n_words_src, n_words_trg):
   iter = 0
   while True:
     if reshuffle:
         os.popen('python shuffle.py '+ source_data + ' ' + target_data)
         os.popen('mv '+ source_data + '.shuf ' + source_data)
         os.popen('mv '+ target_data + '.shuf ' + target_data)
     gen_force_train = TextIterator(source_data, target_data, source_dict, target_dict, batch_size, maxlen, n_words_src, n_words_trg)
     ExampleNum = 0
     EpochStart = time.time()
     for x, y in gen_force_train:
         if len(x) < batch_size and len(y) < batch_size:
                 continue
         ExampleNum += len(x)
         yield x, y, iter
     TimeCost = time.time() - EpochStart
     iter +=1
     print('Seen', ExampleNum, 'generator samples. Time cost is ', TimeCost)

def prepare_data(seqs_x, seqs_y, maxlen=None, n_words_src=30000,
                 n_words=30000, precision='float32'):
    # x: a list of sentences
    lengths_x = [len(s) for s in seqs_x]
    lengths_y = [len(s) for s in seqs_y]

    if maxlen is not None:
        new_seqs_x = []
        new_seqs_y = []
        new_lengths_x = []
        new_lengths_y = []
        for l_x, s_x, l_y, s_y in zip(lengths_x, seqs_x, lengths_y, seqs_y):
            if l_x < maxlen and l_y < maxlen:
                new_seqs_x.append(s_x)
                new_lengths_x.append(l_x)
                new_seqs_y.append(s_y)
                new_lengths_y.append(l_y)
        lengths_x = new_lengths_x
        seqs_x = new_seqs_x
        lengths_y = new_lengths_y
        seqs_y = new_seqs_y

        if len(lengths_x) < 1 or len(lengths_y) < 1:
            return None, None, None, None

    n_samples = len(seqs_x)
    maxlen_x = numpy.max(lengths_x) + 1
    maxlen_y = numpy.max(lengths_y) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    y = numpy.zeros((maxlen_y, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(precision)
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype(precision)
    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:lengths_x[idx], idx] = s_x
        x_mask[:lengths_x[idx]+1, idx] = 1.
        y[:lengths_y[idx], idx] = s_y
        y_mask[:lengths_y[idx]+1, idx] = 1.

    return x, x_mask, y, y_mask


def dis_three_length_prepare(seqs_x, seqs_y, seqs_xs, maxlen=50):
    n_samples = len(seqs_x)
    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    y = numpy.zeros((2, n_samples)).astype('int32')
    xs = numpy.zeros((maxlen, n_samples)).astype('int32')
    
    for idx, [s_x, s_y, s_xs] in enumerate(zip(seqs_x, seqs_y, seqs_xs)):
        x[:len(s_x), idx] = s_x
        y[:len(s_y), idx] = s_y
        xs[:len(s_xs), idx] = s_xs
    return x, y, xs

def dis_length_prepare(seqs_x, seqs_y, maxlen=50):
    n_samples = len(seqs_x)
    x = numpy.zeros((maxlen, n_samples)).astype('int32')
    y = numpy.zeros((2, n_samples)).astype('int32')

    for idx, [s_x, s_y] in enumerate(zip(seqs_x, seqs_y)):
        x[:len(s_x), idx] = s_x
        y[:len(s_y), idx] = s_y
    return x, y

def prepare_single_sentence(seqs_x, maxlen=50):
    n_samples = len(seqs_x)
    lens_x = [len(seq) for seq in seqs_x]
    maxlen_x = numpy.max(lens_x) + 1

    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    for idx, s_x in enumerate(seqs_x):
      x[:len(s_x), idx] = s_x
    return x
    
def prepare_multiple_sentence(seqs_x, maxlen=50, precision='float32'):
    n_samples = len(seqs_x)
    lens_x = [len(seq) for seq in seqs_x]
    maxlen_x = numpy.max(lens_x) + 1
    
    x = numpy.zeros((maxlen_x, n_samples)).astype('int32')
    x_mask = numpy.zeros((maxlen_x, n_samples)).astype(precision)

    for idx, s_x in enumerate(seqs_x):
            x[:len(s_x), idx] = s_x
            x_mask[:len(s_x), idx] = 1.

    return x, x_mask

def prepare_sentence_to_maxlen(seqs_x, maxlen=50, precision='float32'):
    n_samples = len(seqs_x)
    x = numpy.zeros((maxlen, n_samples)).astype('int32')

    for idx, s_x in enumerate(seqs_x):
            x[:len(s_x), idx]=s_x
    return x

def extend_sentence_to_maxlen(seqs, maxlen = 50):
    n_samples = len(seqs)
    x=numpy.zeros((n_samples, maxlen)).astype('int32')
    for idx, seq in enumerate(seqs):
        x[idx, :len(seq)]=seq
    return x


def deal_generated_y_sentence(seqs_y, worddicts, precision='float32'):
    n_samples = len(seqs_y)
    lens_y = [len(seq) for seq in seqs_y]
    maxlen_y = numpy.max(lens_y)
    eosTag = '<EOS2>'
    eosIndex = worddicts[1][eosTag]

    y = numpy.zeros((maxlen_y, n_samples)).astype('int32')
    y_mask = numpy.zeros((maxlen_y, n_samples)).astype(precision)

    for idy, s_y in enumerate(seqs_y):
            try:
               firstIndex = s_y.tolist().index(eosIndex)+1
            except ValueError:
               firstIndex = maxlen_y - 1

            y[:firstIndex, idy]=s_y[:firstIndex]
            y_mask[:firstIndex, idy]=1.

    return y, y_mask

def deal_generated_samples(y_sample, dicts):

    eosTag='</S>'
    eosIndex = dicts.get(eosTag)
    #print("eosIndex is", eosIndex)
    n_samples = len(y_sample)
    lens_y = [len(y) for y in y_sample]
    maxlen_y = numpy.max(lens_y)

    y = numpy.zeros((n_samples, maxlen_y)).astype('int32')
    y_mask = numpy.zeros((n_samples, maxlen_y)).astype('float32')

    for idy, s_y in enumerate(y_sample):
         try:
             firstIndex = s_y.tolist().index(eosIndex)  ### </s> not included
         except:
             firstIndex = len(s_y)
         y[idy, :firstIndex]=s_y[:firstIndex]
         y_mask[idy, :firstIndex]=1.

    return y, y_mask

def deal_generated_samples_to_maxlen(y_sample, dicts, maxlen):

    eosTag='</S>'
    eosIndex = dicts.get(eosTag)
    #print("eosIndex is", eosIndex)
    n_samples = len(y_sample)

    y = numpy.zeros((n_samples, maxlen)).astype('int32')
    y_mask = numpy.zeros((n_samples, maxlen)).astype('float32')

    for idy, s_y in enumerate(y_sample):
         try:
             firstIndex = s_y.tolist().index(eosIndex)  ### </s> not included
         except:
             firstIndex = len(s_y)
         y[idy, :firstIndex]=s_y[:firstIndex]
         y_mask[idy, :firstIndex]=1.

    return y, y_mask


def remove_pad_tolist(seqs):
    seqs_removed_list=[]

    for ids, s_y in enumerate(seqs):
        try:
            firstIndex = s_y.tolist().index(0)
        except ValueError:
            firstIndex = len(s_y) - 1
        seqs_removed_list.append(s_y[:firstIndex])
    return seqs_removed_list
        
def ortho_weight(ndim, precision='float32'):
    W=numpy.random.randn(ndim, ndim)
    u,s,v=numpy.linalg.svd(W)
    return u.astype(precision)

def norm_weight(nin, nout=None, scale=0.01, ortho=True, precision='float32'):
    if nout is None:
        nout=nin
    if nout == nin and ortho:
        W=ortho_weight(nin)
    else:
        W=scale * numpy.random.randn(nin,nout)
    return W.astype(precision)

def tableLookup(vocab_size, embedding_size, scope="tableLookup", init_device='/cpu:0', reuse_var=False, prefix='tablelookup'):
   
    if not scope:
        scope=tf.get_variable_scope()

    with tf.variable_scope(scope) as vs:
        if not  reuse_var:
            with tf.device(init_device):
                embeddings_init=norm_weight(vocab_size, embedding_size)
                embeddings=tf.get_variable('embeddings',shape=[vocab_size, embedding_size], initializer=tf.constant_initializer(embeddings_init))
        else:
                tf.get_variable_scope().reuse_variables()
                embeddings=tf.get_variable('embeddings')
    return embeddings
                
def FCLayer(state_below, input_size, output_size,  is_3d = True, reuse_var = False, use_bias=True, activation=None, scope='ff', init_device='/cpu:0', prefix='ff', precision='float32'):
    
    if not scope:
        scope=tf.get_variable_scope()
    
    with tf.variable_scope(scope):
        if not reuse_var:
            with tf.device(init_device):
                W_init = norm_weight(input_size, output_size)
                matrix=tf.get_variable('W', [input_size, output_size], initializer=tf.constant_initializer(W_init), trainable=True)
                if use_bias:
                    bias_init =  numpy.zeros((output_size,)).astype(precision)
                    bias = tf.get_variable('b', output_size, initializer=tf.constant_initializer(bias_init), trainable=True)
        else:
            tf.get_variable_scope().reuse_variables()
            matrix=tf.get_variable('W')
            if use_bias:
                bias=tf.get_variable('b')

        inputShape = tf.shape(state_below)
        if is_3d :
            state_below=tf.reshape(state_below, [-1, inputShape[2]])
            output=tf.matmul(state_below, matrix)
            output=tf.reshape(output, [-1, inputShape[1] , output_size])
        else :
            output=tf.matmul(state_below, matrix)
        if use_bias:
            output=tf.add(output, bias)
        if activation is not None:
            output = activation(output)
    return output


def average_clip_gradient(tower_grads, clip_c):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)
			#Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)
			# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)
		# Keep in mind that the Variables are redundant because they are shared
		#  across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	if clip_c > 0:
			grad, value = zip(*average_grads)
			grad, global_norm = tf.clip_by_global_norm(grad, clip_c)
			average_grads = zip(grad,value)
			
	#self.average_grads = average_grads
	
	return average_grads

def average_clip_gradient_by_value(tower_grads, clip_min, clip_max):
	average_grads = []
	for grad_and_vars in zip(*tower_grads):
		# Note that each grad_and_vars looks like the following:
		#   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
		grads = []
		for g, _ in grad_and_vars:
			# Add 0 dimension to the gradients to represent the tower.
			expanded_g = tf.expand_dims(g, 0)
			#Append on a 'tower' dimension which we will average over below.
			grads.append(expanded_g)
			# Average over the 'tower' dimension.
		grad = tf.concat(axis=0, values=grads)
		grad = tf.reduce_mean(grad, 0)
		# Keep in mind that the Variables are redundant because they are shared
		#  across towers. So .. we will just return the first tower's pointer to
		# the Variable.
		v = grad_and_vars[0][1]
		grad_and_var = (grad, v)
		average_grads.append(grad_and_var)
	if clip_max > 0:
			grad, value = zip(*average_grads)
			grad = [tf.clip_by_value(x, clip_min, clip_max) for x in grad]
			average_grads = zip(grad,value)
			
	#self.average_grads = average_grads
	
	return average_grads


def get_ngrams(input_tokens, max_n=None):
    if max_n is None:
        max_n = 4

    n_grams=[]
    for n in range(1, max_n+1):
        n_grams.append(defaultdict(int))
        for n_gram in zip(*[input_tokens[i:] for i in range(n)]):
            n_grams[n-1][n_gram] +=1
    return n_grams

def score(ref_tokens, hypothesis_tokens, max_n=None):
    if max_n is None:
        max_n =4

    def product(iterable):
        return reduce(mul, iterable, 1)

    def n_gram_precision(ref_ngrams, hyp_ngrams):
        precision=[]
        for n in range(1, max_n + 1):
            overlap = 0
            for ref_ngram, ref_ngram_count in ref_ngrams[n-1].iteritems():
                if ref_ngram in hyp_ngrams[n-1]:
                    overlap += min(ref_ngram_count, hyp_ngrams[n-1][ref_ngram])
            hyp_length = max(0, len(hypothesis_tokens)-n+1)
            if n >=2:
                overlap += 1
                hyp_length += 1
            precision.append(overlap/hyp_length if hyp_length > 0 else 0.0)
        return precision

    def brevity_penalty(ref_length, hyp_length):
        return min(1.0, exp(1-(ref_length/hyp_length if hyp_length > 0 else 0.0)))

    hypothesis_length = len(hypothesis_tokens)
    ref_length = len(ref_tokens)
    hypothesis_ngrams = get_ngrams(hypothesis_tokens)
    ref_ngrams = get_ngrams(ref_tokens)

    np = n_gram_precision(ref_ngrams, hypothesis_ngrams)
    bp = brevity_penalty(ref_length, hypothesis_length)

    return product(np)**(1 / max_n) * bp


