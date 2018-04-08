import tensorflow as tf
import numpy 
import copy
import sys
import time
import random
import pdb
import cPickle as pkl
import os

from collections import OrderedDict
from six.moves import xrange,zip

from data_iterator import TextIterator
from data_iterator import genTextIterator
from data_iterator import disTextIterator
from data_iterator import fopen

from cnn_discriminator import DisCNN

from gru_cell import GRULayer
from gru_cell import GRUCondLayer

from gru_cell import RANLayer
from gru_cell import RANCondLayer

from share_function import _p
from share_function import prepare_data
from share_function import dis_length_prepare
from share_function import ortho_weight
from share_function import norm_weight
from share_function import tableLookup
from share_function import FCLayer
from share_function import average_clip_gradient
from share_function import prepare_single_sentence
from share_function import prepare_multiple_sentence
from share_function import gen_train_iter
from share_function import print_string
from share_function import deal_generated_y_sentence

from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.ops import variable_scope as vs

from tensorflow.contrib.rnn import GRUCell, RNNCell
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import rnn
from tensorflow.python.ops import variable_scope
from tensorflow.python.ops.rnn import dynamic_rnn

logging.set_verbosity(logging.INFO)

class GenNmt(object):
    def __init__(self, sess, batch_size, source_dict, target_dict, train_data_source, train_data_target, 
                     n_words_src, n_words_trg, gpu_device, dim_word=512, dim = 1024, max_len=30, clip_c = 1.0,  max_epoches=10, reshuffle=False, saveto='./', saveFreq=400, dispFreq=10, sampleFreq=1, 
                     optimizer='adadelta', precision='float32', gen_reload = False, DebugMode=False ):
          self.sess = sess
          
          self.batch_size = batch_size
          
          logging.info('Load dictionary ')
          # load dictionaries and invert them
          dictionaries=[]
          dictionaries.append(source_dict)
          dictionaries.append(target_dict)
          worddicts = [None] * len(dictionaries)
          worddicts_r = [None] * len(dictionaries)
          for ii, dd in enumerate(dictionaries):
            with open(dd, 'rb') as f:
                worddicts[ii] = pkl.load(f)
                worddicts_r[ii] = dict()
                for kk, vv in worddicts[ii].iteritems():
                   worddicts_r[ii][vv] = kk
          
          self.dictionaries = dictionaries
          self.worddicts = worddicts
          self.worddicts_r = worddicts_r
          logging.info('done ')
          
          
          logging.info('Parser traing params')
          
          # parser precision
          if precision == 'float32':
            self.dtype = 'float32'
            #self.dtype = tf.float32
          elif precision == 'float16':
            self.dtype = 'float16'
            #self.dtype = tf.float16
          
          # parser devices
          gpu_string = gpu_device
          gpu_devices = []
          gpu_devices = gpu_string.split('-')
          self.gpu_devices = gpu_devices[1:]
          self.gpu_num = len(self.gpu_devices)

          self.train_data_source = train_data_source
          self.train_data_target = train_data_target
          self.n_words_src = n_words_src
          self.n_words_trg = n_words_trg
          self.max_len = max_len
          self.dim_word = dim_word
          self.dim = dim
          self.precision = precision
          self.gen_reload = gen_reload
          self.DebugMode = DebugMode
          self.clip_c = clip_c
          self.max_epoches = max_epoches
          self.reshuffle = reshuffle
          self.saveto = saveto
          self.saveFreq = saveFreq
          self.dispFreq = dispFreq
          self.sampleFreq = sampleFreq

          if optimizer == 'adam':
            self.optimizer = tf.train.AdamOptimizer()
          elif optimizer == 'adadelta':
            self.optimizer = tf.train.AdadeltaOptimizer(learning_rate=1.,rho=0.95,epsilon=1e-6)
          elif optimizer == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(0.0001)
          elif optimizer == 'rmsprop':
            self.optimizer = tf.train.RMSPropOptimizer()
          else :
            raise ValueError("optimizer must be adam, adadelta or sgd.")
          logging.info('done')
          
          # Build placeholder
          logging.info('Build placeholder ')
          self.build_placeholder()
          logging.info('done')
          
          logging.info( 'Build data iterator')
          self.train_it = self.train_iter()
          logging.info('done')
          
          self.reuse_var = False
          logging.info('Model init done ')

          #print('build train model')
          #self.build_train_model()
          #print('done')

          #params = [param for param in tf.all_variables()]
          #if not self.sess.run(tf.is_variable_initialized(params[0])):
          #   init_op = tf.initialize_variables(params)
          #   self.sess.run(init_op)
          #saver = tf.train.Saver(tf.all_variables())
          #self.saver=saver

          #if self.gen_reload:
          #  print('reloading params from %s '% self.saveto)
          #  self.saver.restore(self.sess, self.saveto)
          #  print('reloading params done')

          #print('build generate')
          #self.build_generate()
          #self.rollout_generate()
          #print('done')

        
    def train_iter(self):
        Epoch=0;
        while True:
            if self.reshuffle:
                os.popen('python  shuffle.py  '+self.train_data_source+' '+ self.train_data_target)
                os.popen('mv '+ self.train_data_source+'.shuf   '+ self.train_data_source)
                os.popen('mv '+ self.train_data_target+'.shuf   '+ self.train_data_target)
            train = TextIterator(self.train_data_source, self.train_data_target,
                         self.dictionaries[0], self.dictionaries[1],
                         n_words_source= self.n_words_src, n_words_target= self.n_words_trg,
                         batch_size= self.batch_size * self.gpu_num,
                         maxlen= self.max_len)
            ExamplesNum=0;
            print( 'Epoch : ' , Epoch )
            EpochStart = time.time()
            for x,y in train:
                if len(x) < self.gpu_num * self.batch_size:
                    continue
                ExamplesNum+=len(x);
                yield x, y, Epoch
            TimeCost = time.time() - EpochStart;    
            Epoch+=1;
            print('Seen ',ExamplesNum,' examples. Time Cost : ',TimeCost)

    def build_placeholder(self):
        self.x_list = []
        self.y_list = []
        self.x_mask_list = []
        self.y_mask_list = []
        for i in range(self.gpu_num):
            x = tf.placeholder(tf.int32, [None, self.batch_size])
            x_mask = tf.placeholder(self.dtype, [None, self.batch_size])
            y = tf.placeholder(tf.int32, [None, self.batch_size])
            y_mask = tf.placeholder(self.dtype, [None, self.batch_size])
            
            self.x_list.append(x)
            self.x_mask_list.append(x_mask)
            self.y_list.append(y)
            self.y_mask_list.append(y_mask)
    
    def get_inputs(self, gpu_device):
        try:
            gpu_id = self.gpu_devices.index(gpu_device)
        except:
            logging.warn("get inputs error. input gpu_device : %s .", gpu_device)
            raise ValueError(" get inputs error ! ")
        return self.x_list[gpu_id], self.x_mask_list[gpu_id], self.y_list[gpu_id], self.y_mask_list[gpu_id] 
    
    def build_model(self, reuse_var=False, gpu_device='0'):

        with tf.device("/gpu:%d" % int(gpu_device)):
            #tf.get_variable_scope().reuse_variables()
            logging.info("building model on device %s , reuse : %d.", gpu_device,reuse_var)
            n_samples = self.batch_size
            # ---------------- Inputs & Mask ---------------
            x, x_mask, y, y_mask = self.get_inputs(gpu_device)
            xr = tf.reverse(x, [True, False])
            xr_mask=tf.reverse(x_mask, [True, False])
            
            x_flat = tf.reshape(x, [-1])
            xr_flat = tf.reshape(xr, [-1])
            y_flat = tf.reshape(y, [-1])

            x_mask_tile = tf.tile(x_mask[:,:,None], [1, 1, self.dim])
            y_mask_tile = tf.tile(y_mask[:,:,None], [1, 1, self.dim])
            xr_mask_tile = tf.tile(xr_mask[:,:,None], [1, 1, self.dim])

            # ------------------ Embedding -----------------
            # embedding
            sourcetable = tableLookup(self.n_words_src, self.dim_word, scope = 'sourceTable', 
                                        reuse_var=reuse_var, prefix='Wemb')
            emb = tf.nn.embedding_lookup(sourcetable, x_flat)
            emb = tf.reshape(emb, [ -1, n_samples, self.dim_word])
            
            embr = tf.nn.embedding_lookup(sourcetable, xr_flat)
            embr = tf.reshape(embr, [-1, n_samples, self.dim_word])
            
            # ------------------- Encoder ------------------
            # forward encode
            cellForward = GRULayer(self.dim, scope='encoder_f', input_size=self.dim_word, 
                                    activation=math_ops.tanh, prefix = 'encoder', precision=self.precision, reuse_var=reuse_var)
            proj = dynamic_rnn(cellForward, (emb,x_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_f', swap_memory=True)
            contextForward = proj[0]
            # backward encode
            cellBackward = GRULayer(self.dim, scope='encoder_b', input_size=self.dim_word, 
                                    activation=math_ops.tanh, prefix = 'encoder_r', precision=self.precision,reuse_var=reuse_var)
            projr = dynamic_rnn(cellBackward, (embr,xr_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_b', swap_memory=True)
            contextBackward = projr[0]
            
            # get init state for decoder
            ctx = tf.concat( axis=2, values=[contextForward, contextBackward[::-1]] )
            ctx_mean = tf.reduce_sum(tf.multiply(ctx, x_mask[:,:,None]),0) / tf.reduce_sum(x_mask, 0)[:, None]
            init_state=FCLayer(ctx_mean, self.dim*2, self.dim, is_3d=False, 
                                reuse_var=reuse_var, scope='ff_state', prefix='ff_state', precision=self.precision)
            
            # ------------------ Decoder -----------------
            targettable= tableLookup(self.n_words_trg, self.dim_word, scope='targetTable', 
                                                    reuse_var=reuse_var, prefix='Wemb_dec')
            emb_y = tf.nn.embedding_lookup(targettable, y_flat)
            emb_y = tf.reshape(emb_y, [-1, n_samples, self.dim_word])
            n_timesteps_trg = tf.shape(emb_y)[0]
            emb_y=tf.concat(axis=0, values=[ tf.zeros([1, n_samples, self.dim_word]), 
                                tf.slice(emb_y, [0,0,0],[n_timesteps_trg-1, n_samples, self.dim_word]) ] )

            cellDecoder =  GRUCondLayer(self.dim, ctx, scope='decoder', context_mask=x_mask, input_size=self.dim_word, 
                                        prefix='decoder', precision=self.precision, reuse_var=reuse_var)
            cellDecoder.set_pctx_()
            proj = dynamic_rnn(cellDecoder, (emb_y,y_mask_tile), initial_state=init_state , 
                                time_major=True, dtype=tf.float32, scope='decoder', swap_memory=True)
            
            states = tf.slice(proj[0], [0, 0, 0], [n_timesteps_trg, n_samples, self.dim])
            ctxs = tf.slice(proj[0], [0, 0, self.dim], [n_timesteps_trg, n_samples, self.dim * 2])
            
            # ------------ Logit & Softmax & Cost -------------
            logit_rnn=FCLayer(states, self.dim, self.dim_word,
                                reuse_var = reuse_var, scope = 'ff_logit_lstm', prefix = 'ff_logit_lstm', precision=self.precision)
            logit_prev=FCLayer(emb_y, self.dim_word, self.dim_word,
                                reuse_var = reuse_var, scope = 'ff_logit_prev', prefix = 'ff_logit_prev', precision=self.precision)
            logit_ctx=FCLayer(ctxs, self.dim*2, self.dim_word,
                                reuse_var = reuse_var, scope = 'ff_logit_ctx', prefix = 'ff_logit_ctx', precision=self.precision)

            logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
            logit = FCLayer(logit, self.dim_word, self.n_words_trg,
                                reuse_var = reuse_var, scope = 'ff_logit', prefix = 'ff_logit', precision=self.precision)
            logit = tf.reshape(logit, [ -1, tf.shape(logit)[2] ]) 
            cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=y_flat)
            cost = tf.reshape(cost, [-1, n_samples])
            
            cost = tf.reduce_sum( cost * y_mask , 0)
            cost = tf.reduce_mean(cost, 0 )
            # -------------------- grad --------------------

            grad = self.optimizer.compute_gradients(cost)
            
            self.sourcetable=sourcetable
            self.targettable=targettable
            self.cellForward = cellForward  
            self.cellBackward = cellBackward
            self.cellDecoder = cellDecoder
            
            # ----------------- Debug Print ----------------
            if self.DebugMode:
                tensorPrint={}
                tensorPrint['embsrc'] = emb
                tensorPrint['embsrcr'] = embr
                tensorPrint['ctx'] = ctx
                tensorPrint['init_state'] = init_state
                tensorPrint['decoder_states'] = states
                tensorPrint['decoder_ctxs'] = ctxs
                tensorPrint['grad'] = grad
                self.tensorPrint = tensorPrint
                # for inner tensors print
                #self.cellForward = cellForward  
                #self.cellBackward = cellBackward
                #self.cellDecoder = cellDecoder
            return  cost, grad

    def build_sampler(self, gpu_device='/gpu:0'):
            reuse_var = False
            with tf.device(gpu_device):
                
                x = tf.placeholder(tf.int32, [None, 1])
                x_mask = tf.ones_like(x, dtype = self.precision)
                
                xr = tf.reverse(x, [True, False])
                xr_mask=tf.ones_like(xr, dtype = self.precision)
            
                x_flat = tf.reshape(x, [-1])
                xr_flat = tf.reshape(xr, [-1])

                x_mask_tile = tf.tile(x_mask[:,:,None], [1,1,self.dim])
                xr_mask_tile = tf.tile(xr_mask[:,:,None], [1,1,self.dim])

                sourcetable = tableLookup(self.n_words_src, self.dim_word, scope = 'sourceTable', 
                                        reuse_var=reuse_var, prefix='Wemb')
                emb = tf.nn.embedding_lookup(sourcetable, x_flat)
                emb = tf.reshape(emb, [ -1,1,self.dim_word])
            
                embr = tf.nn.embedding_lookup(sourcetable, xr_flat)
                embr = tf.reshape(embr, [-1,1, self.dim_word])
            
                # ------------------- Encoder ------------------
                # forward encode
                cellForward = GRULayer(self.dim, scope='encoder_f', input_size=self.dim_word, 
                                        activation=math_ops.tanh, prefix = 'encoder', reuse_var=reuse_var)
                proj = dynamic_rnn(cellForward, (emb,x_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_f', swap_memory=True)
                contextForward = proj[0]
                # backward encode
                cellBackward = GRULayer(self.dim, scope='encoder_b', input_size=self.dim_word, 
                                        activation=math_ops.tanh, prefix = 'encoder_r', reuse_var=reuse_var)
                projr = dynamic_rnn(cellBackward, (embr,xr_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_b', swap_memory=True)
                contextBackward = projr[0]
            
                # get init state for decoder
                ctx = tf.concat( axis=2, values=[contextForward, contextBackward[::-1]] )
                ctx_mean = tf.reduce_mean( ctx , 0) 
                init_state=FCLayer(ctx_mean, self.dim*2, self.dim, is_3d=False, activation=math_ops.tanh,
                                    reuse_var=reuse_var, scope='ff_state', prefix='ff_state')

                # ------------------- One Step Decoder ------------------
                self.state = tf.placeholder( self.precision , [ None, self.dim])
                self.context = tf.placeholder( self.precision , [ None, None, self.dim * 2])
                self.y = tf.placeholder( tf.int32 , [None])
                
                
                targettable= tableLookup(self.n_words_trg, self.dim_word, scope='targetTable', 
                                                        reuse_var=reuse_var, prefix='Wemb_dec')
                
                def f1(): return tf.zeros( [  1, self.dim_word] )
                def f2(): return tf.nn.embedding_lookup(targettable, self.y)
                emb_y = control_flow_ops.cond(tf.less(self.y[0], 0), f1, f2)
                emb_y = tf.reshape(emb_y, [-1, self.dim_word])
                n_beam = tf.shape(emb_y)[0]
                
                cellDecoder =  GRUCondLayer(self.dim, self.context,  scope='decoder', context_mask = x_mask, input_size=self.dim_word, 
                                            prefix='decoder', reuse_var=reuse_var)
                cellDecoder.set_pctx_()
                with vs.variable_scope('decoder'):
                    proj = cellDecoder( (emb_y, tf.ones([ n_beam, self.dim])), self.state )
                
                
                next_state = tf.slice(proj[0], [0, 0], [ n_beam, self.dim])
                ctxs = tf.slice(proj[0], [0, self.dim], [ n_beam, self.dim * 2])
                
                # ------------ Logit & Softmax & Cost -------------
                logit_rnn=FCLayer(next_state, self.dim, self.dim_word, is_3d=False,
                                    reuse_var = reuse_var, scope = 'ff_logit_lstm', prefix = 'ff_logit_lstm')
                logit_prev=FCLayer(emb_y, self.dim_word, self.dim_word, is_3d=False, 
                                    reuse_var = reuse_var, scope = 'ff_logit_prev', prefix = 'ff_logit_prev')
                logit_ctx=FCLayer(ctxs, self.dim*2, self.dim_word, is_3d=False,
                                    reuse_var = reuse_var, scope = 'ff_logit_ctx', prefix = 'ff_logit_ctx')

                logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
                logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d=False,
                                    reuse_var = reuse_var, scope = 'ff_logit', prefix = 'ff_logit')
                logit = tf.reshape(logit, [ -1, tf.shape(logit)[1] ])
                probs = tf.nn.softmax( logit )
                self.next_p = probs
                self.next_state = next_state
                self.ctx = ctx
                return x, init_state
    
    def gen_sample(self, decode_file, decode_result_file, beam_size, is_print = True, gpu_device='/gpu:0'):
            with tf.device(gpu_device):
                x, init_state = self.build_sampler(gpu_device)
                
                init_op = tf.global_variables_initializer()
                init_local_op = tf.local_variables_initializer()
                saver=tf.train.Saver(tf.global_variables())
                self.sess.run(init_op)
                self.sess.run(init_local_op)
                saver.restore(self.sess, './'+self.saveto)
                DecodeStart = time.time()
                with open(decode_file) as f, open(decode_result_file, 'w') as fo:
                    for idx, line in enumerate(f):
                        #print 'Sentence num: ', idx
                        sample = []
                        sample_score = []

                        live_k = 1
                        dead_k = 0

                        hyp_samples = [[]] * live_k
                        hyp_scores = numpy.zeros(live_k).astype('float32')
                        hyp_states = []

                        # get initial state of decoder rnn and encoder context

                        words = line.strip().split()
                        words = [self.worddicts[0][w] if w in self.worddicts[0] else 1
                                  for w in words]
                        words = [w if w < self.n_words_src else 1 for w in words]
                        words = numpy.array(words + [0] ,dtype = 'int32')[:,None]  
                        next_state, ctx0 =  self.sess.run([init_state, self.ctx] , feed_dict = {x:words})
                        
                        next_w = -1 * numpy.ones((1,)).astype('int32')  # bos indicator
                        
                        for ii in xrange(self.max_len):
                            ctx = numpy.tile(ctx0, [live_k, 1])
                            next_p, next_state  =  self.sess.run([self.next_p, self.next_state ] , feed_dict = {self.state:next_state,
                                                                         self.context:ctx, self.y: next_w, x:words})
                            #pdb.set_trace();


                            cand_scores = hyp_scores[:, None] - numpy.log(next_p)
                            cand_flat = cand_scores.flatten()
                            ranks_flat = cand_flat.argsort()[:(beam_size-dead_k)]

                            voc_size = next_p.shape[1]
                            trans_indices = ranks_flat / voc_size
                            word_indices = ranks_flat % voc_size
                            costs = cand_flat[ranks_flat]
                            
                            new_hyp_samples = []
                            new_hyp_scores = numpy.zeros(beam_size-dead_k).astype('float32')
                            new_hyp_states = []

                            for idx, [ti, wi] in enumerate(zip(trans_indices, word_indices)):
                                new_hyp_samples.append(hyp_samples[ti]+[wi])
                                new_hyp_scores[idx] = copy.copy(costs[idx])
                                new_hyp_states.append(copy.copy(next_state[ti]))

                            # check the finished samples
                            new_live_k = 0
                            hyp_samples = []
                            hyp_scores = []
                            hyp_states = []

                            for idx in xrange(len(new_hyp_samples)):
                                if new_hyp_samples[idx][-1] == 0:
                                    sample.append(new_hyp_samples[idx])
                                    sample_score.append(new_hyp_scores[idx])
                                    dead_k += 1
                                else:
                                    new_live_k += 1
                                    hyp_samples.append(new_hyp_samples[idx])
                                    hyp_scores.append(new_hyp_scores[idx])
                                    hyp_states.append(new_hyp_states[idx])
                            hyp_scores = numpy.array(hyp_scores)
                            live_k = new_live_k

                            if new_live_k < 1:
                                break
                            if dead_k >= beam_size:
                                break

                            next_w = numpy.array([w[-1] for w in hyp_samples])
                            next_state = numpy.array(hyp_states)
                            #for hsample in hyp_samples:
                            #           print(hsample)
                            #print(hyp_scores,'\n')
                        if live_k > 0:
                                for idx in xrange(live_k):
                                    sample.append(hyp_samples[idx])
                                    sample_score.append(hyp_scores[idx])
                        sample_best = sample[ numpy.array(sample_score).argmin() ]
                        sample_str = print_string( 'y', sample_best, self.worddicts_r )
                        if is_print:
                            print sample_str
                        fo.write(sample_str+'\n')
                        #pdb.set_trace();
                        #return sample, sample_score
                #print 'All Decode Time : ', time.time() - DecodeStart
            
    def build_test(self, maxlen=30, reuse_var=True, gpu_device='/gpu:0'):
        with tf.device(gpu_device):
            
            x=tf.placeholder(tf.int32, [None, 1])
            
            x_mask = tf.ones_like(x, dtype=self.precision)

            n_sample = 1

            xr = tf.reverse(x, [True, False])
            xr_mask = tf.ones_like(xr, dtype=self.precision)

            x_flat = tf.reshape(x, [-1])
            xr_flat = tf.reshape(xr, [-1])

            x_mask_tile = tf.tile(x_mask[:,:,None], [1,1,self.dim])
            xr_mask_tile = tf.tile(xr_mask[:,:,None], [1,1,self.dim])

            sourcetable = tableLookup(self.n_words_src, self.dim_word, scope='sourceTable', reuse_var = reuse_var, prefix='Wemb')

            emb = tf.nn.embedding_lookup(sourcetable, x_flat)
            emb = tf.reshape(emb, [-1, n_sample, self.dim_word])

            embr = tf.nn.embedding_lookup(sourcetable, xr_flat)
            embr = tf.reshape(embr, [-1, n_sample, self.dim_word])

            cellForward = GRULayer(self.dim, scope='encoder_f', input_size = self.dim_word, activation=math_ops.tanh, prefix='encoder', reuse_var = reuse_var)
            proj = dynamic_rnn(cellForward, (emb, x_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_f', swap_memory=True)

            contextForward = proj[0]

            cellBackward = GRULayer(self.dim, scope='encoder_b', input_size=self.dim_word, activation=math_ops.tanh, prefix='encoder_r', reuse_var = reuse_var)
            projr = dynamic_rnn(cellBackward, (embr, xr_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_b', swap_memory=True)
            
            contextBackward = projr[0]

            ctx = tf.concat(axis=2, values=[contextForward, contextBackward[::-1]])

            ctx_mean = tf.reduce_mean(ctx, 0)

            init_state = FCLayer(ctx_mean, self.dim*2, self.dim, is_3d=False, activation=math_ops.tanh, reuse_var = reuse_var, scope='ff_state', prefix='ff_state', precision=self.precision)

            cellDecoder_s = GRUCondLayer(self.dim, ctx, scope='decoder', context_mask=x_mask, input_size=self.dim_word, prefix='decoder', precision=self.precision, reuse_var=reuse_var)
            
            cellDecoder_s.set_pctx_()

            targettable = tableLookup(self.n_words_trg, self.dim_word, scope='targetTable', reuse_var=reuse_var, prefix='Wemb_dec')

            g_prediction = tensor_array_ops.TensorArray(dtype=tf.float32, size=maxlen, dynamic_size=True, infer_shape=True)

            y_sample = tensor_array_ops.TensorArray(dtype=tf.int64, size=maxlen, dynamic_size=True, infer_shape=True)

            y0 = tf.zeros([1, self.dim_word])
            
            with tf.variable_scope('decoder'):
                proj_y = cellDecoder_s((y0, tf.ones([1, self.dim])), init_state)

            next_state = tf.slice(proj_y[0], [0, 0], [1, self.dim])
            ctxs = tf.slice(proj_y[0], [0, self.dim], [1, self.dim * 2])

            logit_rnn = FCLayer(next_state, self.dim, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_lstm', prefix='ff_logit_lstm')
            logit_prev =FCLayer(y0, self.dim_word, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_prev', prefix='ff_logit_prev')
            logit_ctx = FCLayer(ctxs, self.dim * 2, self.dim_word, is_3d = False, reuse_var = reuse_var, scope='ff_logit_ctx',  prefix='ff_logit_ctx')
            
            logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
            logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d = False, reuse_var = reuse_var, scope='ff_logit', prefix='ff_logit')
            logit = tf.reshape(logit, [-1, self.n_words_trg])

            next_probs = tf.nn.softmax(logit)
            print('the shape of the next_probs ', next_probs.get_shape())
            next_max = tf.argmax(next_probs, 1)
            log_probs = tf.log(next_probs)
            print('the shape of the log_probs ', log_probs.get_shape())

            next_log_sample = tf.multinomial(log_probs, 1)
            next_sample = tf.multinomial(next_probs, 1)

            next_sample_flat = tf.reshape(next_sample, [-1])
            next_sample_flat = next_sample_flat[0]

            #y_sample = y_sample.write(i, next_sample_flat)
            

            self.test_x = x
            self.test_max = next_max
            self.test_sample = next_sample
            self.test_log_sample = next_log_sample


    def self_test(self, infile, outfile):
         infile = fopen(infile, 'r')
         outfile = fopen(outfile, 'w')

         with open(self.dictionaries[0]) as f:
            gen_dict = pkl.load(f)
             
         lines = infile.readlines()
        
         for line in lines:
           #print line

           if line =="":
             continue
           source = []
           line = line.strip().split()

           ll = [gen_dict[w] if w in gen_dict else 1 for w in line]
           ll = [w if w < self.n_words_src else 1 for w in ll]

           if len(ll) > self.max_len:
             continue
           
           source.append(ll)
           ll = prepare_single_sentence(source)

           feed = {self.test_x:ll}
           print(self.sess.run([self.test_max, self.test_sample, self.test_log_sample], feed_dict=feed))

    ## this is the version for batch_size >= 1 

    def build_generate(self, maxlen=50, generate_batch=2, reuse_var=True, optimizer = None, gpu_device='/gpu:0'):
        with tf.device(gpu_device):

            n_sample = generate_batch

            x=tf.placeholder(tf.int32, [None, n_sample])
            x_mask = tf.placeholder(self.dtype, [None, n_sample])

            reward = tf.placeholder(tf.float32, [n_sample, None])
            y = tf.placeholder(tf.int32, [None, n_sample])
            y_mask = tf.placeholder(self.dtype, [None, n_sample])
            
            xr = tf.reverse(x, [True, False])
            xr_mask = tf.reverse(x_mask, [True, False])

            x_flat = tf.reshape(x, [-1])
            xr_flat = tf.reshape(xr, [-1])

            x_mask_tile = tf.tile(x_mask[:,:,None], [1,1,self.dim])
            xr_mask_tile = tf.tile(xr_mask[:,:,None], [1,1,self.dim])

            sourcetable = tableLookup(self.n_words_src, self.dim_word, scope='sourceTable', reuse_var = reuse_var, prefix='Wemb')

            emb = tf.nn.embedding_lookup(sourcetable, x_flat)
            emb = tf.reshape(emb, [-1, n_sample, self.dim_word])

            embr = tf.nn.embedding_lookup(sourcetable, xr_flat)
            embr = tf.reshape(embr, [-1, n_sample, self.dim_word])

            cellForward = GRULayer(self.dim, scope='encoder_f', input_size = self.dim_word, activation=math_ops.tanh, prefix='encoder', reuse_var = reuse_var)
            proj = dynamic_rnn(cellForward, (emb, x_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_f', swap_memory=True)

            contextForward = proj[0]

            cellBackward = GRULayer(self.dim, scope='encoder_b', input_size=self.dim_word, activation=math_ops.tanh, prefix='encoder_r', reuse_var = reuse_var)
            projr = dynamic_rnn(cellBackward, (embr, xr_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_b', swap_memory=True)
            
            contextBackward = projr[0]

            ctx = tf.concat(axis=2, values=[contextForward, contextBackward[::-1]])

            ctx_mean = tf.reduce_sum(tf.multiply(ctx, x_mask[:,:,None]), 0) / tf.reduce_sum(x_mask, 0)[:,None]

            init_state = FCLayer(ctx_mean, self.dim*2, self.dim, is_3d=False, activation=math_ops.tanh, reuse_var = reuse_var, scope='ff_state', prefix='ff_state', precision=self.precision)

            cellDecoder_s = GRUCondLayer(self.dim, ctx, scope='decoder', context_mask=x_mask, input_size=self.dim_word, prefix='decoder', precision=self.precision, reuse_var=reuse_var)
            
            cellDecoder_s.set_pctx_()

            targettable = tableLookup(self.n_words_trg, self.dim_word, scope='targetTable', reuse_var=reuse_var, prefix='Wemb_dec')

            y_sample = tensor_array_ops.TensorArray(dtype=tf.int64, size=maxlen, dynamic_size=True, infer_shape=True)

            def recurrency(i, y, emb_y, init_state, y_sample):
                 with tf.variable_scope('decoder'):
                     proj_y = cellDecoder_s((emb_y, tf.ones([n_sample, self.dim])), init_state)

                 next_state = tf.slice(proj_y[0], [0, 0], [n_sample, self.dim])
                 ctxs = tf.slice(proj_y[0], [0, self.dim], [n_sample, self.dim * 2])

                 logit_rnn = FCLayer(next_state, self.dim, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_lstm', prefix='ff_logit_lstm')
                 logit_prev =FCLayer(emb_y, self.dim_word, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_prev', prefix='ff_logit_prev')
                 logit_ctx = FCLayer(ctxs, self.dim * 2, self.dim_word, is_3d = False, reuse_var = reuse_var, scope='ff_logit_ctx',  prefix='ff_logit_ctx')
                 
                 logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
                 logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d = False, reuse_var = reuse_var, scope='ff_logit', prefix='ff_logit')
                 logit = tf.reshape(logit, [-1, self.n_words_trg])

                 next_probs = tf.nn.softmax(logit)

                 next_sample = tf.argmax(next_probs, 1)
                 
                 y_sample = y_sample.write(i, next_sample)
                 next_emb = tf.nn.embedding_lookup(targettable, next_sample)

                 return i+1, next_sample, next_emb, next_state, y_sample
            
            y0 = tf.zeros([n_sample, self.dim_word])

            index_i = tf.constant(0, dtype=tf.int32)
            index_j = tf.constant(-1, shape=[n_sample], dtype=tf.int64)

            i, _, _, _, y_sample= tf.while_loop(
                    cond = lambda i, j , _2, _3, _4: i < maxlen,
                    body = recurrency,
                    loop_vars = (index_i, index_j, y0, init_state, y_sample),
                    shape_invariants=(index_i.get_shape(), index_j.get_shape(), y0.get_shape(), tf.TensorShape([None, self.dim]), tf.TensorShape([])))

            y_sample = y_sample.pack()
            y_sample = tf.transpose(y_sample, perm=[1, 0])
            
            ## if the sample len is less than 1, make it as 'eos1'
            #eos = '<EOS1>'
            #eosIndex = self.worddicts[1][eos]
            #def f1():
            #    return tf.convert_to_tensor([[eosIndex]], dtype=tf.int64)
            #def f2():
            #    return y_sample

            #y_sample = tf.cond(tf.less(sample_len, 1), f1, f2)


       ###################  for update the parameters #############################################################################
            g_prediction = tensor_array_ops.TensorArray(dtype=tf.float32, size=maxlen, dynamic_size=True, infer_shape=True)

            y_input_array = tensor_array_ops.TensorArray(dtype=tf.int32, size=maxlen, dynamic_size=True, infer_shape = True)

            y_input_array = y_input_array.unpack(y) 

            def gan_recurrency(i, next_y, emb_y, init_state, g_prediction):
                 
                 with tf.variable_scope('decoder'):
                     proj_y = cellDecoder_s((emb_y, tf.ones([n_sample, self.dim])), init_state)

                 next_state = tf.slice(proj_y[0], [0, 0], [n_sample, self.dim])
                 ctxs = tf.slice(proj_y[0], [0, self.dim], [n_sample, self.dim * 2])

                 logit_rnn = FCLayer(next_state, self.dim, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_lstm', prefix='ff_logit_lstm')
                 logit_prev =FCLayer(emb_y, self.dim_word, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_prev', prefix='ff_logit_prev')
                 logit_ctx = FCLayer(ctxs, self.dim * 2, self.dim_word, is_3d = False, reuse_var = reuse_var, scope='ff_logit_ctx',  prefix='ff_logit_ctx')
                 
                 logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
                 logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d = False, reuse_var = reuse_var, scope='ff_logit', prefix='ff_logit')
                 logit = tf.reshape(logit, [-1, self.n_words_trg])

                 next_probs = tf.nn.softmax(logit)
                 
                 next_y = y_input_array.read(i)

                 #next_y = tf.reshape(next_y, [-1])[0]

                 next_emb = tf.nn.embedding_lookup(targettable, next_y)

                 #next_emb = tf.reshape(next_emb, [1, self.dim_word])

                 g_prediction=g_prediction.write(i, next_probs)

                 return i+1, next_y, next_emb, next_state, g_prediction


            gan_y0 = tf.zeros([n_sample, self.dim_word])
            gan_index_i = tf.constant(0, dtype=tf.int32)
            gan_index_j = tf.constant(-1, shape=[n_sample], dtype=tf.int32)


            #y_input_len = tf.shape(y)[0]

            gan_i, _, _, _, g_prediction = tf.while_loop(
                    cond = lambda gan_i, j , _2, _3, _4: tf.less(gan_i,  maxlen),
                    body = gan_recurrency,
                    loop_vars = (gan_index_i, gan_index_j, gan_y0, init_state, g_prediction),
                    shape_invariants=(gan_index_i.get_shape(), gan_index_j.get_shape(), gan_y0.get_shape(), tf.TensorShape([None, self.dim]), tf.TensorShape([])))

           # gan_indices = tf.range(y_input_len)
           # g_predictions = g_prediction.gather(gan_indices)
            g_predictions = g_prediction.pack()
            g_predictions = tf.transpose(g_predictions, perm =[1, 0, 2])
 
            y_transpose = tf.transpose(y, perm=[1, 0])

            g_loss = -tf.reduce_sum( 
                    tf.reduce_sum(
                    tf.one_hot(tf.to_int32(tf.reshape(y_transpose, [-1])), self.n_words_trg, 1.0, 0.0) * tf.log(
                        tf.clip_by_value(tf.reshape(g_predictions, [-1, self.n_words_trg]),1e-20, 1.0)), 1) * tf.reshape(reward, [-1])
                    ,0) / n_sample

            params = [param for param in tf.trainable_variables() if 'generate' in param.name]

            #for param in params:
            #        print param.name
            
            if optimizer == 'adam':
                    print('using adam as the optimizer when build generate')
                    gen_optimizer = tf.train.AdamOptimizer()
            else :
                    print('using the default optimizer of the model when build generate')
                    gen_optimizer = self.optimizer

            g_grad = gen_optimizer.compute_gradients(g_loss, params)

            
            #for grad, var in g_grad:
            #        print var.name, grad

            g_optm = gen_optimizer.apply_gradients(g_grad)

            self.generate_x = x
            self.generate_x_mask = x_mask
            self.generate_y_sample = y_sample

            self.generate_reward = reward
            self.generate_input_y = y
            self.generate_y_mask = y_mask
            self.generate_g_predictions = g_predictions
            self.generate_g_loss = g_loss
            self.generate_g_grad = g_grad
            self.generate_g_optm = g_optm

            return x, reward, y_sample, g_loss, g_grad

    def rollout_generate(self, generate_batch = 2, reuse_var=True):
            
            n_sample = generate_batch
            
            x = tf.placeholder(tf.int32, [None, n_sample])
            x_mask = tf.placeholder(self.dtype, [None, n_sample])

            y = tf.placeholder(tf.int32, [None, n_sample])

            give_num = tf.placeholder(tf.int32, shape=[], name='give_num')

            xr = tf.reverse(x, [True, False])
            xr_mask = tf.reverse(x_mask, [True, False])

            x_flat = tf.reshape(x, [-1])
            xr_flat = tf.reshape(xr, [-1])

            x_mask_tile = tf.tile(x_mask[:,:,None], [1,1,self.dim])
            xr_mask_tile = tf.tile(xr_mask[:,:,None], [1,1,self.dim])

            sourcetable = tableLookup(self.n_words_src, self.dim_word, scope='sourceTable', reuse_var=reuse_var, prefix='Wemb')

            emb = tf.nn.embedding_lookup(sourcetable, x_flat)
            emb = tf.reshape(emb, [-1, n_sample, self.dim_word])

            embr = tf.nn.embedding_lookup(sourcetable, xr_flat)
            embr = tf.reshape(embr, [-1, n_sample, self.dim_word])

            cellForward = GRULayer(self.dim, scope='encoder_f', input_size=self.dim_word, activation=math_ops.tanh, prefix='encoder', reuse_var=reuse_var)
            proj = dynamic_rnn(cellForward, (emb, x_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_f', swap_memory=True)

            contextForward = proj[0]

            cellBackward = GRULayer(self.dim, scope='encoder_b', input_size=self.dim_word, activation=math_ops.tanh, prefix='encoder_r', reuse_var=reuse_var)
            projr = dynamic_rnn(cellBackward, (embr,xr_mask_tile), time_major=True, dtype=tf.float32, scope='encoder_b', swap_memory=True)
            
            contextBackward = projr[0]

            ctx = tf.concat(axis=2, values=[contextForward, contextBackward[::-1]])

            ctx_mean = tf.reduce_sum(tf.multiply(ctx, x_mask[:,:,None]), 0) / tf.reduce_sum(x_mask, 0)[:,None]

            init_state = FCLayer(ctx_mean, self.dim*2, self.dim, is_3d=False, activation=math_ops.tanh, reuse_var = reuse_var, scope='ff_state', prefix='ff_state', precision=self.precision)

            cellDecoder_s = GRUCondLayer(self.dim, ctx, scope='decoder', context_mask=x_mask, input_size=self.dim_word, prefix='decoder', precision=self.precision, reuse_var=reuse_var)
            
            cellDecoder_s.set_pctx_()

            targettable = tableLookup(self.n_words_trg, self.dim_word, scope='targetTable', reuse_var=reuse_var, prefix='Wemb_dec')

            sample_len = self.max_len

            y_sample = tensor_array_ops.TensorArray(dtype=tf.int32, size=sample_len, dynamic_size=True, infer_shape=True)

            y_index = tensor_array_ops.TensorArray(dtype=tf.int32, size=sample_len, dynamic_size=True, infer_shape=True)
            
            y_index = y_index.unpack(y)

            def recurrency_given(i, y, emb_y, given_num, init_state, y_sample):
                with tf.variable_scope('decoder'):
                    proj_y = cellDecoder_s((emb_y, tf.ones([n_sample, self.dim])), init_state)

                next_state = tf.slice(proj_y[0], [0, 0], [n_sample, self.dim])
                ctxs = tf.slice(proj_y[0], [0, self.dim], [n_sample, self.dim * 2])

                logit_rnn = FCLayer(next_state, self.dim, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_lstm', prefix='ff_logit_lstm', precision=self.precision)
                logit_prev = FCLayer(emb_y, self.dim_word, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_prev', prefix='ff_logit_prev', precision=self.precision)
                logit_ctx = FCLayer(ctxs, self.dim*2, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_ctx', prefix = 'ff_logit_ctx', precision=self.precision)
                
                logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
                logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d=False, reuse_var = reuse_var, scope='ff_logit', prefix = 'ff_logit', precision=self.precision)
                logit = tf.reshape(logit, [-1, self.n_words_trg])
                
                next_y = y_index.read(i)
                next_y_e = next_y[:,None]
                next_emb = tf.nn.embedding_lookup(targettable, next_y)
                y_sample = y_sample.write(i, next_y)
              
                return i+1, next_y_e, next_emb, give_num, next_state, y_sample

            def recurrency(i, y, emb_y, give_num, init_state, y_sample):

                #print('the dtype of i is ', i.dtype)
                #print('the shape of y is ', y.dtype)
                with tf.variable_scope('decoder'):
                    proj_y = cellDecoder_s((emb_y, tf.ones([n_sample, self.dim])), init_state)

                next_state = tf.slice(proj_y[0], [0, 0], [n_sample, self.dim])
                ctxs = tf.slice(proj_y[0], [0, self.dim], [n_sample, self.dim * 2])

                logit_rnn = FCLayer(next_state, self.dim, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_lstm', prefix='ff_logit_lstm', precision=self.precision)
                logit_prev = FCLayer(emb_y, self.dim_word, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_prev', prefix='ff_logit_prev', precision=self.precision)
                logit_ctx = FCLayer(ctxs, self.dim*2, self.dim_word, is_3d=False, reuse_var = reuse_var, scope='ff_logit_ctx', prefix = 'ff_logit_ctx', precision=self.precision)
                logit = tf.tanh(logit_rnn + logit_prev + logit_ctx)
                logit = FCLayer(logit, self.dim_word, self.n_words_trg, is_3d=False, reuse_var = reuse_var, scope='ff_logit', prefix = 'ff_logit', precision=self.precision)
                logit = tf.reshape(logit, [-1, self.n_words_trg])
                
                next_probs = tf.nn.softmax(logit)
                log_probs = tf.log(next_probs)
                next_sample = tf.multinomial(log_probs, 1)
                
                next_sample_flat = tf.cast(next_sample, tf.int32) #convert to tf.int32
                next_sample_squeeze = tf.squeeze(next_sample_flat, [1])
                
                y_sample = y_sample.write(i, next_sample_squeeze)
                next_emb = tf.nn.embedding_lookup(self.targettable, next_sample_squeeze)

                return i+1, next_sample_flat, next_emb, give_num, next_state, y_sample

            y0 = tf.zeros([n_sample, self.dim_word])

            i, y_out, emb_y, give_num_out, init_state, y_sample = tf.while_loop(
                    cond = lambda i, _1, _2, give_num, _4, _5: i < give_num,
                    body = recurrency_given,
                    loop_vars = (tf.constant(0, dtype=tf.int32), tf.constant(-1, shape=[n_sample, 1], dtype=tf.int32), y0, give_num, init_state, y_sample),
                    shape_invariants=(tf.TensorShape([]), tf.TensorShape([n_sample, 1]), y0.get_shape(), give_num.get_shape(), tf.TensorShape([None, self.dim]), tf.TensorShape([])))

            _, _, _, _, _, y_sample = tf.while_loop(
                    cond = lambda i, _1, _2, _3, _4, _5 : i < self.max_len,
                    body = recurrency,
                    loop_vars = (i, y_out, emb_y, give_num_out, init_state, y_sample),
                    shape_invariants=(i.get_shape(), tf.TensorShape([n_sample, 1]), emb_y.get_shape(), give_num.get_shape(), tf.TensorShape([None, self.dim]), tf.TensorShape([])))

            y_sample = y_sample.pack()
            
            self.roll_x = x
            self.roll_x_mask = x_mask
            self.roll_y = y
            self.roll_give_num = give_num
            self.roll_y_sample = y_sample

            return x, y, give_num, y_sample

    def init_and_reload(self):

         ##########
         # this function is only used for the gan training with reload
         ##########

         params = [param for param in tf.trainable_variables() if 'generate' in param.name]
         #params = [param for param in tf.all_variables()]
         if not self.sess.run(tf.is_variable_initialized(params[0])):
            #init_op = tf.initialize_variables(params)
            init_op = tf.global_variables_initializer()  ## this is important here to initialize_all_variables()
            self.sess.run(init_op)

         saver = tf.train.Saver(params)
         self.saver=saver
         
         if self.gen_reload:                                     ##here must be true
           print('reloading params from %s '% self.saveto)
           self.saver.restore(self.sess, './'+self.saveto)
           print('reloading params done')
         else:
           print('error, reload must be true!!')


    def generate_step(self, sentence_x, x_mask):
         feed = {self.generate_x:sentence_x, self.generate_x_mask:x_mask}
         y_out  = self.sess.run(self.generate_y_sample, feed_dict = feed)
         return y_out

    def generate_step_and_update(self, sentence_x, x_mask, sentence_y, reward):
         feed = {self.generate_x:sentence_x, self.generate_x_mask:x_mask, self.generate_input_y: sentence_y, self.generate_reward:reward}
         #g_predic = self.sess.run(self.generate_g_predictions, feed_dict=feed)
         loss, _, _ = self.sess.run([self.generate_g_loss, self.generate_g_grad, self.generate_g_optm], feed_dict=feed)
         return loss

    def generate_and_save(self, infile, outfile, generate_batch=2):
         gen_train_it = gen_train_iter(infile, None, self.dictionaries[0], self.n_words_src, generate_batch, self.max_len)
         epoch = 0
         outfile = fopen(outfile, 'w')
         while epoch < 1:
             x, epoch = next(gen_train_it)
             x, x_mask = prepare_multiple_sentence(x)
             feed = {self.generate_x:x, self.generate_x_mask:x_mask}
             y_out = self.sess.run(self.generate_y_sample, feed_dict=feed)
             y_out_dealed, _ = deal_generated_y_sentence(y_out, self.worddicts)
             y_out = numpy.transpose(y_out_dealed)
             
             for id, y in enumerate(y_out):
                     y_str = print_string('y', y, self.worddicts_r)
                     outfile.write(y_str+'\n')

        
    def get_reward(self, x, x_mask, y_sample, y_sample_mask,  rollnum, discriminator):
         rewards= []
         for i in range(rollnum):
            for give_num in numpy.arange(1, self.max_len, dtype='int32'):
               
               feed = {self.roll_x :x, self.roll_x_mask:x_mask, self.roll_y:y_sample, self.roll_give_num:give_num}
               output = self.sess.run(self.roll_y_sample, feed_dict=feed)

               output = output * y_sample_mask
               #print('the shape of the y_sample is ', y_sample.shape)
               #print('the shape of output is ', output.shape)
               
               #samper_str = print_string('y', y_sample[:,0], self.worddicts_r)
               #output_str = print_string('y', output[:,0], self.worddicts_r)
               #
               #samper_str_2 = print_string('y', y_sample[:,1], self.worddicts_r)
               #output_str_2 = print_string('y', output[:,1], self.worddicts_r)
               ##print('i is %d give_num is %d' %(i, give_num))
               #print samper_str
               #print output_str


               #print samper_str_2
               #print output_str_2

               feed = {discriminator.dis_input_x:output,  discriminator.dis_dropout_keep_prob:1.0}
               ypred_for_auc = self.sess.run(discriminator.dis_ypred_for_auc, feed_dict=feed)
               #print('ypred_for_auc is ', ypred_for_auc)
               #print('\n')
               ypred = numpy.array([item[1] for item in ypred_for_auc])
               if i == 0:
                  rewards.append(ypred) 
               else:
                  rewards[give_num - 1]+=ypred

            #print('the shape of the y_sample is ', y_sample.shape)
            #y_sample_len_norm = numpy.zeros((self.max_len, 1)).astype('int32')
            #y_sample_len_norm[:len(y_sample), 0] = y_sample[:,0]

            feed = {discriminator.dis_input_x : y_sample, discriminator.dis_dropout_keep_prob:1.0}
            ypred_for_auc = self.sess.run(discriminator.dis_ypred_for_auc, feed_dict=feed)
            ypred = numpy.array([item[1] for item in ypred_for_auc])

            if i == 0:
               rewards.append(ypred)
            else:
               rewards[self.max_len-1] += ypred
         
         rewards = rewards * y_sample_mask
         rewards = numpy.transpose(numpy.array(rewards)) / (1.0 * rollnum)  
         #print('the shape of reward is ', rewards.shape)
         return rewards    
               
    def build_train_model(self):
        reuse_var = self.reuse_var
        loss=tf.convert_to_tensor(0.)
        grads = []
        for i, gpu_device in enumerate(self.gpu_devices):
                if i > 0 :
                    reuse_var = True
                cost, grad = self.build_model(reuse_var=reuse_var, gpu_device=gpu_device)
                loss += cost
                grads.append(grad)
                
        loss = loss / self.gpu_num
        grads_and_vars = average_clip_gradient(grads, self.clip_c)
        optm = self.optimizer.apply_gradients(grads_and_vars)

        self.train_loss = loss
        self.train_grads_and_vars = grads_and_vars
        self.train_optm = optm

    def gen_train(self):
        
      #  logging.info('Building graph ')
      #  reuse_var = self.reuse_var
      #  
      #  loss=tf.convert_to_tensor(0.)
      #  grads = []
      #  for i, gpu_device in enumerate(self.gpu_devices):
      #          if i > 0 :
      #              reuse_var = True
      #          cost, grad = self.build_model(reuse_var=reuse_var, gpu_device=gpu_device)
      #          loss += cost
      #          grads.append(grad)
      #          
      #  loss = loss / self.gpu_num
      #  grads_and_vars = average_clip_gradient(grads, self.clip_c)
      #  optm = self.optimizer.apply_gradients(grads_and_vars)
      #  
        init_op = tf.global_variables_initializer()
        init_local_op = tf.local_variables_initializer()
        saver=tf.train.Saver(tf.global_variables())
        self.sess.run(init_op)
        self.sess.run(init_local_op)
        logging.info('done')
        
        if self.gen_reload:
            print('reloading params from %s' % self.saveto)
            saver.restore(self.sess, './'+self.saveto)
            print('params reload done')
        
        # time log
        TrainStart = time.time()
        cost_history = []
        word_count_history = []
        epoch = 0
        uidx = 0
        HourIdx = 0
        gpu_num = self.gpu_num
        print('Train begin ')
        while epoch < self.max_epoches:
            # running time log
            if time.time() - TrainStart > 3600 * HourIdx :
                print('-------------------------Hour : %d -------------------------' % HourIdx)
                HourIdx+=1
            
            # prepare inputs
            BatchStart = time.time()
            x, y, epoch = next(self.train_it)
                        
            uidx += 1
            if not len(x) % gpu_num == 0:
                logging.info('pass the last batch')
                continue
            word_count = 0

            x_data_list = numpy.split(numpy.array(x),gpu_num)
            y_data_list = numpy.split(numpy.array(y),gpu_num)
                        
            x_sample=0
            y_sample=0

            myFeed_dict={}
            for i, x, y in zip(range(gpu_num), x_data_list, y_data_list):
                x = x.tolist()
                y = y.tolist()
                x, x_mask, y, y_mask = prepare_data(x,y)
                word_count += y_mask.sum()
                                
                if i == 0:
                    x_sample = x;
                    y_sample = y;

                myFeed_dict[self.x_list[i]]= x
                myFeed_dict[self.x_mask_list[i]]= x_mask
                myFeed_dict[self.y_list[i]]= y
                myFeed_dict[self.y_mask_list[i]]= y_mask

            _, cost = self.sess.run([self.train_optm,self.train_loss], feed_dict=myFeed_dict)
            
            BatchTime = time.time() - BatchStart
            # check and logging
            if numpy.isnan(cost) or numpy.isinf(cost):
                if numpy.isnan(cost):
                    logging.warn('NaN detected')
                else:
                    logging.warn('Inf detected') 
                return 1., 1., 1.

            if numpy.mod(uidx, self.saveFreq) == 0:
                print("----------------------save at epoch %d----------------------" % epoch)
                saver.save(self.sess, self.saveto)

            cost_history.append(cost * self.batch_size * gpu_num)
            word_count_history.append(word_count)
            
            history_num = 100
            MovAve = numpy.sum(cost_history[-history_num:])/numpy.sum(word_count_history[-history_num:])
            
            # display
            if numpy.mod(uidx, self.dispFreq) == 0:
                print('epoch %d Update in this GPU %d sample %d Cost %f MovAveCost %f BatchTime %f ' % (epoch,uidx, uidx*gpu_num*self.batch_size, cost, MovAve, BatchTime))

                        # sample 
                        #if numpy.mod(uidx, self.sampleFreq) == 0:

                        #    for jj in xrange(numpy.minimum(5, x_sample.shape[1])):
                        #        stochastic = False
                        #        sample, score, _, _ = self.gen_sample(x_sample[:,jj][:,None], k=10, maxlen=30, stochastic=stochastic, argmax=False)

                        #        print ('score', jj, ' :  ')
                        #        for vv in x_sample[:, jj]:
                        #            if vv == 0:
                        #                break
                        #            if vv in self.worddicts_r[0]:
                        #                print (self.worddicts_r[0][vv])
                        #            else:
                        #                print('UNK ')

                        #        print()
                        #        
                        #        print ('Truth', jj, ' :  ') 
                        #        for vv in y_sample[:,jj]:
                        #            if vv == 0:
                        #                break
                        #            if vv in self.worddicts_r[1]:
                        #                print (self.worddicts_r[1][vv])
                        #            else:
                        #                print('UNK ')

                        #        print()

                        #        print('Sample', jj, ' :  ')
                        #        if stochastic:
                        #            ss = sample
                        #        else:
                        #            score = score / numpy.array([len(s) for s in sample])
                        #            ss = sample[score.argmin()]
                        #        for vv in ss:
                        #            if vv == 0:
                        #                break
                        #            if vv in self.worddicts_r[1]:
                        #                print (self.worddicts_r[1][vv])
                        #            else:
                        #                print ('UNK ')
                        #        print ("Sample score : ", score.min())
