from __future__ import print_function
import yaml
import time
import os
import sys
import numpy as np
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model
from cnn_discriminator import DisCNN
from share_function import deal_generated_samples
from share_function import deal_generated_samples_to_maxlen
from share_function import extend_sentence_to_maxlen
from share_function import prepare_gan_dis_data
from share_function import FlushFile

def gan_train(config):
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    default_graph=tf.Graph()
    with default_graph.as_default():
        sess = tf.Session(config=sess_config, graph=default_graph)

        logger = logging.getLogger('')
        du = DataUtil(config=config)
        du.load_vocab(src_vocab=config.generator.src_vocab,
                      dst_vocab=config.generator.dst_vocab,
                      src_vocab_size=config.src_vocab_size,
                      dst_vocab_size=config.dst_vocab_size)

        generator = Model(config=config, graph=default_graph, sess=sess)
        generator.build_train_model()
        generator.build_generate(max_len=config.generator.max_length,
                                 generate_devices=config.generator.devices,
                                 optimizer=config.generator.optimizer)

        generator.build_rollout_generate(max_len=config.generator.max_length,
                                         roll_generate_devices=config.generator.devices)

        generator.init_and_restore(modelFile=config.generator.modelFile)

        dis_filter_sizes = [i for i in range(1, config.discriminator.dis_max_len, 4)]
        dis_num_filters = [(100 + i * 10) for i in range(1, config.discriminator.dis_max_len, 4)]

        discriminator = DisCNN(
            sess=sess,
            max_len=config.discriminator.dis_max_len,
            num_classes=2,
            vocab_size=config.dst_vocab_size,
            vocab_size_s=config.src_vocab_size,
            batch_size=config.discriminator.dis_batch_size,
            dim_word=config.discriminator.dis_dim_word,
            filter_sizes=dis_filter_sizes,
            num_filters=dis_num_filters,
            source_dict=config.discriminator.dis_src_vocab,
            target_dict=config.discriminator.dis_dst_vocab,
            gpu_device=config.discriminator.dis_gpu_devices,
            positive_data=config.discriminator.dis_positive_data,
            negative_data=config.discriminator.dis_negative_data,
            source_data=config.discriminator.dis_source_data,
            dev_positive_data=config.discriminator.dis_dev_positive_data,
            dev_negative_data=config.discriminator.dis_dev_negative_data,
            dev_source_data=config.discriminator.dis_dev_source_data,
            max_epoches=config.discriminator.dis_max_epoches,
            dispFreq=config.discriminator.dis_dispFreq,
            saveFreq=config.discriminator.dis_saveFreq,
            saveto=config.discriminator.dis_saveto,
            reload=config.discriminator.dis_reload,
            clip_c=config.discriminator.dis_clip_c,
            optimizer=config.discriminator.dis_optimizer,
            reshuffle=config.discriminator.dis_reshuffle,
            scope=config.discriminator.dis_scope
        )

        batch_iter = du.get_training_batches(
            set_train_src_path=config.generator.src_path,
            set_train_dst_path=config.generator.dst_path,
            set_batch_size=config.generator.batch_size,
            set_max_length=config.generator.max_length
        )

        for epoch in range(1, config.gan_iter_num + 1):
            for gen_iter in range(config.gan_gen_iter_num):
                batch = next(batch_iter)
                x, y_ground = batch[0], batch[1]
                y_sample = generator.generate_step(x)
                logging.info("generate the samples")
                y_sample_dealed, y_sample_mask = deal_generated_samples(y_sample, du.dst2idx)
                #
                #### for debug
                ##print('the sample is ')
                ##sample_str=du.indices_to_words(y_sample_dealed, 'dst')
                ##print(sample_str)
                #
                x_to_maxlen = extend_sentence_to_maxlen(x)
                logging.info("calculate the reward")
                rewards = generator.get_reward(x=x,
                                               x_to_maxlen=x_to_maxlen,
                                               y_sample=y_sample_dealed,
                                               y_sample_mask=y_sample_mask,
                                               rollnum=config.rollnum,
                                               disc=discriminator,
                                               max_len=config.discriminator.dis_max_len,
                                               bias_num=config.bias_num,
                                               data_util=du)

                loss = generator.generate_step_and_update(x, y_sample_dealed, rewards)

                print("the reward is ", rewards)
                print("the loss is ", loss)

                logging.info("save the model into %s" % config.generator.modelFile)
                generator.saver.save(generator.sess, config.generator.modelFile)

                if config.generator.teacher_forcing:

                    logging.info("doiong the teacher forcing begin!")
                    y_ground, y_ground_mask = deal_generated_samples_to_maxlen(
                                                y_sample=y_ground,
                                                dicts=du.dst2idx,
                                                maxlen=config.discriminator.dis_max_len)

                    rewards_ground=np.ones_like(y_ground)
                    rewards_ground=rewards_ground*y_ground_mask
                    loss = generator.generate_step_and_update(x, y_ground, rewards_ground)
                    print("the teacher forcing reward is ", rewards_ground)
                    print("the teacher forcing loss is ", loss)

            generator.saver.save(generator.sess, config.generator.modelFile)

            logging.info("prepare the gan_dis_data begin")
            data_num = prepare_gan_dis_data(
                train_data_source=config.generator.src_path,
                train_data_target=config.generator.dst_path,
                gan_dis_source_data=config.discriminator.dis_source_data,
                gan_dis_positive_data=config.discriminator.dis_positive_data,
                num=config.generate_num,
                reshuf=True
            )

            logging.info("generate and the save in to %s." %config.discriminator.dis_negative_data)
            generator.generate_and_save(data_util=du,
                                        infile=config.discriminator.dis_source_data,
                                        generate_batch=config.discriminator.dis_batch_size,
                                        outfile=config.discriminator.dis_negative_data
                                      )

            logging.info("prepare %d gan_dis_data done!" %data_num)
            logging.info("finetuen the discriminator begin")

            discriminator.train(max_epoch=config.gan_dis_iter_num,
                                positive_data=config.discriminator.dis_positive_data,
                                negative_data=config.discriminator.dis_negative_data,
                                source_data=config.discriminator.dis_source_data
                                )
            discriminator.saver.save(discriminator.sess, discriminator.saveto)
            logging.info("finetune the discrimiantor done!")

        logging.info('reinforcement training done!')

if __name__ == '__main__':
    sys.stdout = FlushFile(sys.stdout)
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.logdir):
        os.makedirs(config.logdir)
    logging.basicConfig(filename=config.logdir+'/train.log', level=logging.INFO)
    console = logging.StreamHandler()
    console.setLevel(logging.DEBUG)
    logging.getLogger('').addHandler(console)
    # Train
    gan_train(config)

