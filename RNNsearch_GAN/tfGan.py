import tensorflow as tf
import numpy 
import copy
import sys
import time
import random
import pdb
import cPickle as pkl

from collections import OrderedDict
from six.moves import xrange,zip

from data_iterator import TextIterator
from data_iterator import genTextIterator
from data_iterator import disTextIterator

from cnn_discriminator import DisCNN
from nmt_generator import GenNmt


from gru_cell import GRULayer
from gru_cell import GRUCondLayer

from share_function import _p
from share_function import prepare_data
from share_function import dis_length_prepare
from share_function import ortho_weight
from share_function import norm_weight
from share_function import tableLookup
from share_function import FCLayer
from share_function import average_clip_gradient
from share_function import gen_train_iter
from share_function import dis_train_iter
from share_function import FlushFile
from share_function import prepare_gan_dis_data
from share_function import prepare_single_sentence
from share_function import prepare_multiple_sentence
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

tf.app.flags.DEFINE_float('train_len', 1, 'train for this many minibatch for synchoronizing')
tf.app.flags.DEFINE_integer('dim_word', 512, 'the dimension of the word embedding')
tf.app.flags.DEFINE_integer('dim', 1024, 'the number of rnn units')
tf.app.flags.DEFINE_integer('patience', 10, 'the patience for early stop')

tf.app.flags.DEFINE_integer('max_epoches', 10000, 'the max epoches for training')
tf.app.flags.DEFINE_integer('dis_epoches', 10, 'the max epoches for training')

tf.app.flags.DEFINE_integer('gan_total_iter_num', 1, 'the max epoches for training')
tf.app.flags.DEFINE_integer('gan_gen_iter_num', 1, 'the max epoches for training')
tf.app.flags.DEFINE_integer('gan_dis_iter_num', 1, 'the max epoches for training')

tf.app.flags.DEFINE_integer('dispFreq', 50, 'train for this many minibatches for displaying')
tf.app.flags.DEFINE_integer('dis_dispFreq', 1, 'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer('gan_dispFreq', 50, 'train for this many minibatches for displaying the gan gen training')
tf.app.flags.DEFINE_integer('dis_saveFreq', 100, 'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer('gan_saveFreq', 100, 'train for this many minibatches for displaying discriminator')
tf.app.flags.DEFINE_integer('dis_devFreq', 100, 'train for this many minibatches for displaying discriminator')

tf.app.flags.DEFINE_integer('validFreq', 1000, 'train for this many minibatches for validation')
tf.app.flags.DEFINE_integer('saveFreq', 2000, 'train for this many minibatches for saving model')
tf.app.flags.DEFINE_integer('sampleFreq', 10000000, 'train for this many minibatches for sampling')

tf.app.flags.DEFINE_float('l2_r', 0.0001, 'L2 regularization penalty')
tf.app.flags.DEFINE_float('lr', 0.1, 'learning rate')
tf.app.flags.DEFINE_float('alpha_c', 0.0, 'alignment regularization')
tf.app.flags.DEFINE_float('clip_c', 5, 'gradient clipping threshold')
tf.app.flags.DEFINE_integer('n_words_src', 30000, 'the size of the source vocabulary')
tf.app.flags.DEFINE_integer('n_words_trg', 30000, 'the size of the target vocabulary')
tf.app.flags.DEFINE_integer('max_len', 30, 'the max length of the training sentence')
tf.app.flags.DEFINE_integer('dis_max_len', 30, 'the max length of the training sentence for discriminator')

tf.app.flags.DEFINE_integer('batch_size', 60, 'the size of the minibatch for training')
tf.app.flags.DEFINE_integer('dis_batch_size', 10, 'the size of the minibatch for training discriminator ')
tf.app.flags.DEFINE_integer('gen_batch_size', 1, 'the size of the minibatch for training generator ')
tf.app.flags.DEFINE_integer('gan_gen_batch_size', 2, 'the size of the minibatch for training generator ')
tf.app.flags.DEFINE_integer('gan_dis_batch_size', 1, 'the size of the minibatch for training generator ')

tf.app.flags.DEFINE_integer('valid_batch_size', 10, 'the size of the minibatch for validation')

tf.app.flags.DEFINE_string('optimizer', 'adadelta', 'the optimizing method')

tf.app.flags.DEFINE_string('saveto', 'nmt1000w', 'the file name used to store the model')
tf.app.flags.DEFINE_string('dis_saveto', 'disriminator', 'the file name used to store the model of the discriminator')

tf.app.flags.DEFINE_string('train_data_source', '/data/zhyang/dl4mt/corpus/data_1000w_golden/source_u8.txt.shuf', 'the train data set of the soruce side')
tf.app.flags.DEFINE_string('train_data_target', '/data/zhyang/dl4mt/corpus/data_1000w_golden/target_u8.txt.shuf', 'the train data set of the target side')

tf.app.flags.DEFINE_string('dis_positive_data', '/data/zhyang/dl4mt/corpus/data_test1000/positive_u8.txt.shuf', 'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string('dis_negative_data', '/data/zhyang/dl4mt/corpus/data_test1000/negative_u8.txt.shuf', 'the negative train data set for the discriminator')

tf.app.flags.DEFINE_string('dis_dev_positive_data', '/data/zhyang/dl4mt/corpus/data_test1000/dev_positive_u8.txt.shuf', 'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string('dis_dev_negative_data', '/data/zhyang/dl4mt/corpus/data_test1000/dev_negative_u8.txt.shuf', 'the negative train data set for the discriminator')


tf.app.flags.DEFINE_string('gan_gen_source_data', '/data/zhyang/dl4mt/corpus/data_test1000/gan_gen_source_u8.txt', 'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string('gan_dis_source_data', '/data/zhyang/dl4mt/corpus/data_test1000/gan_dis_source_data', 'the positive train data set for the discriminator')
tf.app.flags.DEFINE_string('gan_dis_positive_data', '/data/zhyang/dl4mt/corpus/data_test1000/gan_dis_positive_data', 'the negative train data set for the discriminator')
tf.app.flags.DEFINE_string('gan_dis_negative_data', '/data/zhyang/dl4mt/corpus/data_test1000/gan_dis_negative_data', 'the positive train data set for the discriminator')

tf.app.flags.DEFINE_string('valid_data_source', 'data/zhyang/dl4mt/source.txt', 'the valid data set of the soruce size')
tf.app.flags.DEFINE_string('valid_data_target', 'data/zhyang/dl4mt/target.txt', 'the valid data set of the target side')

tf.app.flags.DEFINE_string('source_dict', '/data/zhyang/dl4mt/corpus/data_1000w_golden/source_u8.txt.shuf.pkl', 'the source vocabulary')
tf.app.flags.DEFINE_string('target_dict', '/data/zhyang/dl4mt/corpus/data_1000w_golden/target_u8.txt.shuf.pkl', 'the target vocabulary')

tf.app.flags.DEFINE_boolean('use_dropout', False, 'whether to use dropout')
tf.app.flags.DEFINE_boolean('gen_reload',False, 'whether to reload the generate model from model file')
tf.app.flags.DEFINE_boolean('dis_reload',False, 'whether to reload the discriminator model from model file')

tf.app.flags.DEFINE_boolean('reshuffle', False, 'whether to reshuffle train data')
tf.app.flags.DEFINE_boolean('dis_reshuffle', False, 'whether to reshuffle train data of the discriminator')
tf.app.flags.DEFINE_boolean('gen_reshuffle', False, 'whether to reshuffle train data of the generator')
tf.app.flags.DEFINE_boolean('gan_gen_reshuffle', False, 'whether to reshuffle train data of the generator')
tf.app.flags.DEFINE_boolean('gan_dis_reshuffle', False, 'whether to reshuffle train data of the generator')

tf.app.flags.DEFINE_boolean('DebugMode', False, 'whether to debug')

tf.app.flags.DEFINE_string('gpu_device', 'gpu-0-1-2-3-4-5-6-7', 'this many gpus used to train the model')
tf.app.flags.DEFINE_string('dis_gpu_device', 'gpu-14-15', 'this many gpus used to train the generator model')

tf.app.flags.DEFINE_string('cpu_device', 'cpu-0', 'this cpu used to train the model')
tf.app.flags.DEFINE_string('init_device', '/cpu:0', 'this cpu used to train the model')

tf.app.flags.DEFINE_string('precision', 'float32', 'precision on GPU')

tf.app.flags.DEFINE_integer('rollnum', 16, 'the rollnum for rollout')
tf.app.flags.DEFINE_integer('generate_num', 200000, 'the rollnum for rollout')

tf.app.flags.DEFINE_boolean('is_gan_train', False, 'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean('is_generator_train', False, 'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean('is_discriminator_train', False, 'whether to do generative adversarial train')
tf.app.flags.DEFINE_boolean('is_decode', False, 'whether to decode')
tf.app.flags.DEFINE_boolean('decode_is_print', False, 'whether to decode')
tf.app.flags.DEFINE_string('decode_gpu', '/gpu:0', 'the device used to decode')

tf.app.flags.DEFINE_string('decode_file', 'data/zhyang/dl4mt/source.txt', 'the file to be decoded')
tf.app.flags.DEFINE_string('decode_result_file', 'data/zhyang/dl4mt/result.txt', 'the file to save the decode results')

FLAGS=tf.app.flags.FLAGS

params = OrderedDict()

logging.set_verbosity(logging.INFO)


def main(argv):
            #####################################   create the session  ##################################################################
           
                config = tf.ConfigProto()
                config.gpu_options.allow_growth = True
                config.gpu_options.per_process_gpu_memory_fraction = 1.0
                config.allow_soft_placement = True
                
                is_gan_train = FLAGS.is_gan_train
                is_decode = FLAGS.is_decode
                is_generator_train = FLAGS.is_generator_train
                is_discriminator_train = FLAGS.is_discriminator_train

            ######################################  pretraining  the generator ######################################################################
                batch_size = FLAGS.batch_size
                source_dict = FLAGS.source_dict
                target_dict = FLAGS.target_dict
                train_data_source = FLAGS.train_data_source
                train_data_target = FLAGS.train_data_target
                n_words_src = FLAGS.n_words_src
                n_words_trg = FLAGS.n_words_trg
                gpu_device = FLAGS.gpu_device
                dim_word = FLAGS.dim_word
                dim = FLAGS.dim
                max_len = FLAGS.max_len
                optimizer = FLAGS.optimizer
                precision = FLAGS.precision
                clip_c = FLAGS.clip_c
                max_epoches = FLAGS.max_epoches
                reshuffle = FLAGS.reshuffle
                saveto = FLAGS.saveto
                saveFreq = FLAGS.saveFreq
                dispFreq = FLAGS.dispFreq
                sampleFreq = FLAGS.sampleFreq
                gen_reload = FLAGS.gen_reload
            
                gan_gen_batch_size = FLAGS.gan_gen_batch_size
                
                sess = tf.Session(config=config)
                with tf.variable_scope('generate'):
                    generator = GenNmt(sess = sess, 
                                        batch_size = batch_size, 
                                        source_dict = source_dict, 
                                        target_dict = target_dict, 
                                        train_data_source = train_data_source, 
                                        train_data_target = train_data_target, 
                                        n_words_src = n_words_src, 
                                        n_words_trg = n_words_trg, 
                                        gpu_device = gpu_device, 
                                        dim_word = dim_word, 
                                        dim = dim, 
                                        max_len = max_len, 
                                        clip_c = clip_c, 
                                        max_epoches = max_epoches, 
                                        reshuffle = reshuffle, 
                                        saveto = saveto, 
                                        saveFreq = saveFreq, 
                                        dispFreq = dispFreq, 
                                        sampleFreq = sampleFreq, 
                                        optimizer = optimizer, 
                                        precision = precision,
                                        gen_reload = gen_reload)
              
                    if is_decode:
                                decode_file = FLAGS.decode_file
                                decode_result_file = FLAGS.decode_result_file
                                decode_gpu = FLAGS.decode_gpu
                                decode_is_print = FLAGS.decode_is_print
                                #print('decoding the file %s on %s' % (decode_file, decode_gpu))
                                generator.gen_sample(decode_file, decode_result_file, 10, is_print = decode_is_print, gpu_device=decode_gpu)

                                return 0

                    elif is_generator_train:
                                print('train the model and build the generate')
                                generator.build_train_model()
                                generator.gen_train()
                                generator.build_generate(maxlen=max_len, generate_batch=gan_gen_batch_size, optimizer='adam')
                                generator.rollout_generate(generate_batch=gan_gen_batch_size)
                                print('done')

                    else:
                         print('build the generate without training')
                         generator.build_train_model()
                         generator.build_generate(maxlen=max_len, generate_batch=gan_gen_batch_size,optimizer='adam')
                         generator.rollout_generate(generate_batch=gan_gen_batch_size)
                         generator.init_and_reload()
                         
                         #print('building testing ')
                         #generator.build_test()
                         #print('done')

            
            ## #################################################### pretraining the discriminator ##################################################################

                if is_discriminator_train or is_gan_train:
                
                   dis_max_epoches = FLAGS.dis_epoches
                   dis_dispFreq = FLAGS.dis_dispFreq
                   dis_saveFreq = FLAGS.dis_saveFreq
                   dis_devFreq = FLAGS.dis_devFreq
                   dis_batch_size = FLAGS.dis_batch_size
                   dis_saveto = FLAGS.dis_saveto
                   dis_reshuffle = FLAGS.dis_reshuffle
                   dis_gpu_device = FLAGS.dis_gpu_device
                   dis_max_len = FLAGS.dis_max_len
                   positive_data = FLAGS.dis_positive_data
                   negative_data = FLAGS.dis_negative_data
                   dis_dev_positive_data = FLAGS.dis_dev_positive_data
                   dis_dev_negative_data = FLAGS.dis_dev_negative_data
                   dis_reload = FLAGS.dis_reload
 
                   dis_filter_sizes = [i for i in range(1,dis_max_len, 4)]
                   dis_num_filters = [ (100 + i *10) for i in range(1, dis_max_len, 4)]
                   
                   discriminator = DisCNN(sess = sess, 
                                          max_len = dis_max_len, 
                                          num_classes = 2, 
                                          vocab_size = n_words_trg, 
                                          batch_size = dis_batch_size, 
                                          dim_word = dim_word, 
                                          filter_sizes = dis_filter_sizes, 
                                          num_filters = dis_num_filters, 
                                          source_dict = source_dict, 
                                          target_dict = target_dict, 
                                          gpu_device = dis_gpu_device, 
                                          positive_data = positive_data, 
                                          negative_data = negative_data, 
                                          dev_positive_data = dis_dev_positive_data,
                                          dev_negative_data = dis_dev_negative_data,
                                          max_epoches = dis_max_epoches, 
                                          dispFreq = dis_dispFreq, 
                                          saveFreq = dis_saveFreq,
                                          devFreq = dis_devFreq,
                                          saveto = dis_saveto,
                                          reload = dis_reload,
                                          clip_c = clip_c, 
                                          optimizer = optimizer, 
                                          reshuffle = dis_reshuffle,
                                          scope = 'discnn')

                   if is_discriminator_train:
                      print('train the discriminator')
                      discriminator.train()
                      print('done')

                   else:
                      print('building the discriminator without training done')
                      print('done')
                         
               #   ####################################### Start Reinforcement Training #######################################################################################
                   if is_gan_train:

                      gan_total_iter_num = FLAGS.gan_total_iter_num
                      gan_gen_iter_num = FLAGS.gan_gen_iter_num
                      gan_dis_iter_num = FLAGS.gan_dis_iter_num

                      gan_gen_reshuffle = FLAGS.gan_gen_reshuffle
                      gan_gen_source_data = FLAGS.gan_gen_source_data

                      gan_dis_source_data = FLAGS.gan_dis_source_data
                      gan_dis_positive_data = FLAGS.gan_dis_positive_data
                      gan_dis_negative_data = FLAGS.gan_dis_negative_data
                      gan_dis_reshuffle = FLAGS.gan_dis_reshuffle
                      gan_dis_batch_size = FLAGS.gan_dis_batch_size
                      gan_dispFreq = FLAGS.gan_dispFreq
                      gan_saveFreq = FLAGS.gan_saveFreq
                      roll_num = FLAGS.rollnum
                      generate_num = FLAGS.generate_num
   
                      print('reinforcement training begin...')
                      
                      for gan_iter in range(gan_total_iter_num):
               
                           print('reinforcement training for %d epoch' % gan_iter)
                           gen_train_it = gen_train_iter(gan_gen_source_data, gan_gen_reshuffle, generator.dictionaries[0], n_words_src, gan_gen_batch_size, max_len)
                           
                           print('finetune the generator begin...')
                           for gen_iter in range(gan_gen_iter_num):
                                x, _ = next(gen_train_it)
                                x, x_mask = prepare_multiple_sentence(x, maxlen=max_len)
                                y_sample_out = generator.generate_step(x, x_mask)
                                #print(y_sample_out)
                                #tmp_str=print_string('y', y_sample_out[0], generator.worddicts_r)
                                #tmp_str_2=print_string('y', y_sample_out[1], generator.worddicts_r)
                                #print tmp_str
                                #print tmp_str_2

                                y_input, y_input_mask = deal_generated_y_sentence(y_sample_out, generator.worddicts, precision=precision)
                                rewards = generator.get_reward(x, x_mask, y_input, y_input_mask, roll_num, discriminator)
                                print('the reward is ', rewards)
                                loss = generator.generate_step_and_update(x, x_mask, y_input, rewards)
                                if gen_iter % gan_dispFreq == 0:
                                   print('the %d iter, seen %d examples, loss is %f ' % ( gen_iter, ((gan_iter) * gan_gen_iter_num + gen_iter + 1), loss))
                                if gen_iter % gan_saveFreq == 0:
                                   generator.saver.save(generator.sess, generator.saveto)
                                   print('save the parameters when seen %d examples ' % ((gan_iter) * gan_gen_iter_num + gan_iter + 1))

                           generator.saver.save(generator.sess, generator.saveto)
                           print('finetune the generator done!')
                           
                           #print('self testing')
                           #generator.self_test(gan_dis_source_data, gan_dis_negative_data)
                           #print('self testing done!')
                           
                           print('prepare the gan_dis_data begin ')
                           data_num = prepare_gan_dis_data(train_data_source, train_data_target, gan_dis_source_data, gan_dis_positive_data, num=generate_num, reshuf=True)
                           print('prepare the gan_dis_data done, the num of the gan_dis_data is %d' % data_num)

                           print('generator generate and save to %s' % gan_dis_negative_data)
                           generator.generate_and_save(gan_dis_source_data, gan_dis_negative_data, generate_batch=gan_gen_batch_size)
                           print('done!')
                           
                           print('finetune the discriminator begin...')
                           discriminator.train(max_epoch = gan_dis_iter_num, positive_data=gan_dis_positive_data, negative_data=gan_dis_negative_data)
                           print('finetune the discriminator done!')

                      print('reinforcement training done')
  
if __name__ =='__main__':
    sys.stdout = FlushFile(sys.stdout)
    tf.app.run()
