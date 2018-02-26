from __future__ import print_function
import yaml
import time
import os
import logging
from argparse import ArgumentParser
import tensorflow as tf

from utils import DataUtil, AttrDict
from model import Model


def train(config):
    logger = logging.getLogger('')

    """Train a model with a config file."""
    du = DataUtil(config=config)
    du.load_vocab()

    model = Model(config=config)
    model.build_train_model()

    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess_config.allow_soft_placement = True

    with model.graph.as_default():
        saver = tf.train.Saver(var_list=tf.global_variables())
        summary_writer = tf.summary.FileWriter(config.train.logdir, graph=model.graph)
        # saver_partial = tf.train.Saver(var_list=[v for v in tf.trainable_variables() if 'Adam' not in v.name])

        with tf.Session(config=sess_config) as sess:
            # Initialize all variables.
            sess.run(tf.global_variables_initializer())
            try:
                # saver_partial.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
                # print('Restore partial model from %s.' % config.train.logdir)
                saver.restore(sess, tf.train.latest_checkpoint(config.train.logdir))
            except:
                logger.info('Failed to reload model.')
            for epoch in range(1, config.train.num_epochs+1):
                for batch in du.get_training_batches_with_buckets():
                    start_time = time.time()
                    step = sess.run(model.global_step)
                    # Summary
                    if step % config.train.summary_freq == 0:
                        step, lr, gnorm, loss, acc, summary, _ = sess.run(
                            [model.global_step, model.learning_rate, model.grads_norm,
                             model.loss, model.acc, model.summary_op, model.train_op],
                            feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                        summary_writer.add_summary(summary, global_step=step)
                    else:
                        step, lr, gnorm, loss, acc, _ = sess.run(
                            [model.global_step, model.learning_rate, model.grads_norm,
                             model.loss, model.acc, model.train_op],
                            feed_dict={model.src_pl: batch[0], model.dst_pl: batch[1]})
                    logger.info(
                        'epoch: {0}\tstep: {1}\tlr: {2:.6f}\tgnorm: {3:.4f}\tloss: {4:.4f}\tacc: {5:.4f}\ttime: {6:.4f}'.
                        format(epoch, step, lr, gnorm, loss, acc, time.time() - start_time))

                    # Save model
                    if step % config.train.save_freq == 0:
                        mp = config.train.logdir + '/model_epoch_%d_step_%d' % (epoch, step)
                        saver.save(sess, mp)
                        logger.info('Save model in %s.' % mp)
            logger.info("Finish training.")


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('-c', '--config', dest='config')
    args = parser.parse_args()
    # Read config
    config = AttrDict(yaml.load(open(args.config)))
    # Logger
    if not os.path.exists(config.train.logdir):
        os.makedirs(config.train.logdir)
    logging.basicConfig(filename=config.train.logdir+'/train.log', level=logging.DEBUG)
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    # Train
    train(config)
