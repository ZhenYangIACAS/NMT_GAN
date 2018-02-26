import tensorflow as tf
from tensorflow.python.ops import init_ops
import numpy as np
import logging
import codecs

from tensor2tensor.common_attention import multihead_attention, add_timing_signal_1d, attention_bias_ignore_padding, attention_bias_lower_triangle
from tensor2tensor.common_layers import layer_norm, conv_hidden_relu, smoothing_cross_entropy
from share_function import deal_generated_samples
from share_function import score
from share_function import remove_pad_tolist

INT_TYPE = np.int32
FLOAT_TYPE = np.float32


class Model(object):
    def __init__(self, config, graph=None, sess=None):
        if graph is None:
            self.graph=tf.Graph()
        else:
            self.graph = graph

        if sess is None:
            self.sess=tf.Session(graph=self.graph)
        else:
            self.sess=sess

        self.config = config
        self._logger = logging.getLogger('model')
        self._prepared = False
        self._summary = True

    def prepare(self, is_training):
        assert not self._prepared
        self.is_training = is_training
        # Select devices according to running is_training flag.
        devices = self.config.train.devices if is_training else self.config.test.devices
        self.devices = ['/gpu:'+i for i in devices.split(',')] or ['/cpu:0']
        # If we have multiple devices (typically GPUs), we set /cpu:0 as the sync device.
        self.sync_device = self.devices[0] if len(self.devices) == 1 else '/cpu:0'

        if is_training:
            with self.graph.as_default():
                with tf.device(self.sync_device):
                    # Preparing optimizer.
                    self.global_step = tf.get_variable(name='global_step', dtype=INT_TYPE, shape=[],
                                                       trainable=False, initializer=tf.zeros_initializer)
                    self.learning_rate = tf.convert_to_tensor(self.config.train.learning_rate)
                    if self.config.train.optimizer == 'adam':
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'adam_decay':
                        self.learning_rate = learning_rate_decay(self.config, self.global_step)
                        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate,
                                                                beta1=0.9, beta2=0.98, epsilon=1e-9)
                    elif self.config.train.optimizer == 'sgd':
                        self.optimizer = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate)
                    elif self.config.train.optimizer == 'mom':
                        self.optimizer = tf.train.MomentumOptimizer(self.learning_rate, momentum=0.9)
                    else:
                        logging.info("No optimizer is defined for the model")
                        raise ValueError
        self._initializer = init_ops.variance_scaling_initializer(scale=1, mode='fan_avg', distribution='uniform')
        # self._initializer = tf.uniform_unit_scaling_initializer()
        self._prepared = True

    def build_train_model(self):
        """Build model for training. """
        self.prepare(is_training=True)
        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                acc_list, loss_list, gv_list = [], [], []
                for i, (X, Y, device) in enumerate(zip(Xs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self._logger.info('Build model on %s.' % device)
                        encoder_output = self.encoder(X, reuse=i>0 or None)
                        decoder_output = self.decoder(shift_right(Y), encoder_output, reuse=i > 0 or None)
                        acc, loss = self.train_output(decoder_output, Y, reuse=i > 0 or None)
                        acc_list.append(acc)
                        loss_list.append(loss)
                        gv_list.append(self.optimizer.compute_gradients(loss))

                self.acc = tf.reduce_mean(acc_list)
                self.loss = tf.reduce_mean(loss_list)

                # Clip gradients and then apply.
                grads_and_vars = average_gradients(gv_list)
                if self._summary:
                    for g, v in grads_and_vars:
                        tf.summary.histogram('variables/' + v.name.split(':')[0], v)
                        tf.summary.histogram('gradients/' + v.name.split(':')[0], g)
                grads, self.grads_norm = tf.clip_by_global_norm([gv[0] for gv in grads_and_vars],
                                                                clip_norm=self.config.train.grads_clip)
                grads_and_vars = zip(grads, [gv[1] for gv in grads_and_vars])
                self.train_op = self.optimizer.apply_gradients(grads_and_vars, global_step=self.global_step)

                # Summaries
                tf.summary.scalar('acc', self.acc)
                tf.summary.scalar('loss', self.loss)
                tf.summary.scalar('learning_rate', self.learning_rate)
                tf.summary.scalar('grads_norm', self.grads_norm)
                self.summary_op = tf.summary.merge_all()

    def build_generate(self, max_len, generate_devices, optimizer='rmsprop'):
        with self.graph.as_default():
            with tf.device(self.sync_device):
                if optimizer=='adam':
                    logging.info("using adam for g_loss")
                    optimizer=tf.train.AdamOptimizer(self.config.generator.learning_rate)
                if optimizer=='adadelta':
                    logging.info("using adadelta for g_loss")
                    optimizer=tf.train.AdadeltaOptimizer()
                else:
                    logging.info("using rmsprop for g_loss")
                    optimizer=tf.train.RMSPropOptimizer(self.config.generator.learning_rate)

                src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='gene_src_pl')
                dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='gene_dst_pl')
                reward_pl = tf.placeholder(dtype=tf.float32, shape=[None, None], name='gene_reward')

                generate_devices= ['/gpu:' + i for i in generate_devices.split(',')] or ['/cpu:0']

                Xs = split_tensor(src_pl, len(generate_devices))
                Ys = split_tensor(dst_pl, len(generate_devices))
                Rs = split_tensor(reward_pl, len(generate_devices))

                batch_size_list = [tf.shape(X)[0] for X in Xs]

                encoder_outputs = [None] * len(generate_devices)
                for i, (X, device) in enumerate(zip(Xs, generate_devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self._logger.info('Build generate model on %s' % device)
                        encoder_output = self.encoder(X, reuse=True)
                        encoder_outputs[i] = encoder_output

                def recurrency(i, cur_y, encoder_output):
                    decoder_output=self.decoder(shift_right(cur_y), encoder_output, reuse=True)

                    next_logits = top(body_output=decoder_output,
                                 vocab_size = self.config.dst_vocab_size,
                                 dense_size = self.config.hidden_units,
                                 shared_embedding = self.config.train.shared_embedding,
                                 reuse=True)

                    #with tf.variable_scope("output", initializer=self._initializer, reuse=True):
                    #    next_logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
                    next_logits = next_logits[:, i, :]
                    next_logits = tf.reshape(next_logits, [-1, self.config.dst_vocab_size])
                    next_probs = tf.nn.softmax(next_logits)
                    next_sample = tf.argmax(next_probs, 1)
                    next_sample= tf.expand_dims(next_sample, -1)
                    next_sample = tf.to_int32(next_sample)
                    next_y = tf.concat([cur_y[:, :i],next_sample], axis=1)
                    next_y = tf.pad(next_y, [[0,0], [0, max_len-1-i]])
                    next_y.set_shape([None, max_len])
                    return i+1, next_y, encoder_output


                total_results=[None] * len(generate_devices)
                for i, (device, batch_size) in enumerate(zip(generate_devices, batch_size_list)):
                    with tf.device(lambda op:self.choose_device(op, device)):
                        initial_y = tf.zeros((batch_size, max_len), dtype=INT_TYPE)   ##begin with <s>
                        initial_i = tf.constant(0, dtype=tf.int32)
                        _, sample_result, _ = tf.while_loop(
                            cond = lambda a, _1, _2: a<max_len,
                            body=recurrency,
                            loop_vars= (initial_i, initial_y, encoder_outputs[i]),
                            shape_invariants=(initial_i.get_shape(), initial_y.get_shape(), encoder_outputs[i].get_shape())
                        )
                        total_results[i]=sample_result

                generate_result = tf.concat(total_results, axis=0)

                 #################generate over here ###################################

                loss_list=[]
                grads_and_vars_list=[]
                for i, (Y, reward, device) in enumerate(zip(Ys, Rs, generate_devices)):
                    with tf.device(lambda op:self.choose_device(op, device)):
                        decoder_output=self.decoder(shift_right(Y), encoder_outputs[i], reuse=True)
                        g_loss = self.gan_output(decoder_output, Y, reward, reuse=True)
                        grads_and_vars=optimizer.compute_gradients(g_loss)
                        loss_list.append(g_loss)
                        grads_and_vars_list.append(grads_and_vars)

                grads_and_vars=average_gradients(grads_and_vars_list)
                loss = tf.reduce_mean(loss_list)
                g_optm=optimizer.apply_gradients(grads_and_vars)

                self.generate_x = src_pl
                self.generate_y = dst_pl
                self.generate_reward = reward_pl

                self.generate_sample =generate_result
                self.generate_g_loss = loss
                self.generate_g_grad = grads_and_vars
                self.generate_g_optm = g_optm

    def build_rollout_generate(self, max_len, roll_generate_devices):

        src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='gene_src_pl')
        dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='gene_dst_pl')
        give_num_pl = tf.placeholder(dtype=INT_TYPE, shape=[], name='give_num_pl')

        devices = ['/gpu:' + i for i in roll_generate_devices.split(',')] or ['/cpu:0']
        Xs = split_tensor(src_pl, len(devices))
        Ys = split_tensor(dst_pl, len(devices))

        Ms = [give_num_pl] * len(devices)

        batch_size_list = [tf.shape(X)[0] for X in Xs]

        encoder_outputs = [None] * len(devices)
        for i, (X, device) in enumerate(zip(Xs, devices)):
            with tf.device(lambda op: self.choose_device(op, device)):
                self._logger.info('Build roll generate model on %s' % device)
                encoder_output = self.encoder(X, reuse=True)
                encoder_outputs[i] = encoder_output

        def recurrency(given_num, given_y, encoder_output):
            decoder_output = self.decoder(shift_right(given_y), encoder_output, reuse=True)
            next_logits = top(body_output=decoder_output,
                         vocab_size = self.config.dst_vocab_size,
                         dense_size = self.config.hidden_units,
                         shared_embedding = self.config.train.shared_embedding,
                         reuse=True)
            #with tf.variable_scope("output", initializer=self._initializer, reuse=True):
            #    next_logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            next_logits = next_logits[:, given_num, :]
            #print(next_logits)
            next_probs = tf.nn.softmax(next_logits)
            #print(next_probs)
            log_probs = tf.log(next_probs)
            #print(log_probs)
            next_sample = tf.multinomial(log_probs, 1)
            #print(next_sample)
            next_sample_flat = tf.cast(next_sample, tf.int32)
            next_y = tf.concat([given_y[:, :given_num], next_sample_flat], axis=1)
            next_y = tf.pad(next_y, [[0, 0], [0, max_len - given_num -1]])
            next_y.set_shape([None, max_len])
            return given_num +1, next_y, encoder_output

        total_results = [None] * len(devices)
        for i, (Y, given_num, device) in enumerate(zip(Ys, Ms, devices)):
            with tf.device(lambda op: self.choose_device(op, device)):
                given_y = Y[:, :given_num]

                init_given_y = tf.pad(given_y, [[0, 0], [0, (max_len-given_num)]])
                _, roll_sample, _ = tf.while_loop(
                    cond = lambda a, _1, _2: a < max_len,
                    body=recurrency,
                    loop_vars=(given_num, init_given_y, encoder_outputs[i]),
                    shape_invariants=(given_num.get_shape(), init_given_y.get_shape(), encoder_outputs[i].get_shape())
                )
                total_results[i]=roll_sample

        sample_result = tf.concat(total_results, axis=0)

        self.roll_x = src_pl
        self.roll_y = dst_pl
        self.roll_give_num = give_num_pl
        self.roll_y_sample = sample_result

    def generate_step(self, sentence_x):
        feed={self.generate_x:sentence_x}
        y_sample = self.sess.run(self.generate_sample, feed_dict=feed)
        return y_sample

    def generate_step_and_update(self, sentence_x, sentence_y, reward):
        feed={self.generate_x:sentence_x, self.generate_y:sentence_y, self.generate_reward:reward}
        loss, _, _ = self.sess.run([self.generate_g_loss, self.generate_g_grad, self.generate_g_optm], feed_dict=feed)
        return loss

    def generate_and_save(self, data_util, infile, generate_batch, outfile):
        outfile = codecs.open(outfile, 'w', 'utf-8')
        for batch in data_util.get_test_batches(infile, generate_batch):
            feed={self.generate_x:batch}
            out_generate=self.sess.run(self.generate_sample, feed_dict=feed)
            out_generate_dealed, _ = deal_generated_samples(out_generate, data_util.dst2idx)

            y_strs=data_util.indices_to_words_del_pad(out_generate_dealed, 'dst')
            for y_str in y_strs:
                outfile.write(y_str+'\n')
        outfile.close()

    
    def get_reward(self, x, x_to_maxlen, y_sample, y_sample_mask, rollnum, disc, max_len=50, bias_num=None, data_util=None):
        
        rewards=[]
        x_to_maxlen=np.transpose(x_to_maxlen)

        for i in range(rollnum):
            for give_num in np.arange(1, max_len, dtype='int32'):
                feed={self.roll_x:x, self.roll_y:y_sample, self.roll_give_num:give_num}
                output = self.sess.run(self.roll_y_sample, feed_dict=feed)
                
                #print(output.shape)
                #print(y_sample_mask.shape)

                #print("the sample is ", data_util.indices_to_words(y_sample))
                #print("the roll_sample result is ", data_util.indices_to_words(output))

                output=output * y_sample_mask
                #print("the roll aftter sample_mask is", data_util.indices_to_words(output))
                output=np.transpose(output)

                feed={disc.dis_input_x:output, disc.dis_input_xs:x_to_maxlen,
                      disc.dis_dropout_keep_prob:1.0}
                ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)

                ypred=np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                else:
                    rewards[give_num -1]+=ypred

            y_sample_transpose = np.transpose(y_sample)

            feed = {disc.dis_input_x:y_sample_transpose, disc.dis_input_xs:x_to_maxlen,
                    disc.dis_dropout_keep_prob:1.0}

            ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)
            ypred= np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
            else:
                rewards[max_len -1]+=ypred

        rewards = np.transpose(np.array(rewards)) ## now rewards: batch_size * max_len

        if  bias_num is None:
            rewards = rewards * y_sample_mask  
            rewards = rewards / (1. * rollnum)
        else:
            bias = np.zeros_like(rewards)
            bias +=bias_num * rollnum
            rewards_minus_bias = rewards-bias

            rewards=rewards_minus_bias * y_sample_mask
            rewards = rewards / (1. * rollnum)
        return rewards

    def get_reward_Obinforced(self, x, x_to_maxlen, y_sample, y_sample_mask, y_ground, rollnum, disc, max_len=50, bias_num=None, data_util=None, namana=0.7):
        rewards = []
        BLEU = []

        y_ground_removed_pad_list = remove_pad_tolist(y_ground)
        
        x_to_maxlen=np.transpose(x_to_maxlen)
        y_sample_transpose = np.transpose(y_sample)

        for i in range(rollnum):
            for give_num in np.arange(1, max_len, dtype='int32'):
                feed={self.roll_x:x, self.roll_y:y_sample, self.roll_give_num:give_num}
                output = self.sess.run(self.roll_y_sample, feed_dict=feed)
                
                output=output * y_sample_mask
                output_removed_pad_list = remove_pad_tolist(output)
                #print("the roll aftter sample_mask is", data_util.indices_to_words(output))
                output=np.transpose(output)

                feed={disc.dis_input_x:output, disc.dis_input_xs:x_to_maxlen,
                      disc.dis_dropout_keep_prob:1.0}
                ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)

                BLEU_predict = []
                for hypo_tokens, ref_tokens in zip(output_removed_pad_list, y_ground_removed_pad_list):
                    BLEU_predict.append(score(ref_tokens, hypo_tokens))
                BLEU_predict = np.array(BLEU_predict)

                ypred=np.array([item[1] for item in ypred_for_auc])
                if i == 0:
                    rewards.append(ypred)
                    BLEU.append(BLEU_predict)
                else:
                    rewards[give_num -1]+=ypred
                    BLEU[give_num -1]+=BLEU_predict

            
            #y_sample_transpose = np.transpose(y_sample)

            feed = {disc.dis_input_x:y_sample_transpose, disc.dis_input_xs:x_to_maxlen,
                    disc.dis_dropout_keep_prob:1.0}

            ypred_for_auc=self.sess.run(disc.dis_ypred_for_auc, feed_dict=feed)
            
            y_sample_removed_pad_list = remove_pad_tolist(y_sample)
            BLEU_predict=[]
            for hypo_tokens, ref_tokens in zip(y_sample_removed_pad_list, y_ground_removed_pad_list):
                BLEU_predict.append(score(ref_tokens, hypo_tokens))
            BLEU_predict = np.array(BLEU_predict)

            ypred= np.array([item[1] for item in ypred_for_auc])

            if i == 0:
                rewards.append(ypred)
                BLEU.append(BLEU_predict)
            else:
                rewards[max_len -1]+=ypred
                BLEU[max_len -1]+=BLEU_predict

        rewards = np.transpose(np.array(rewards)) ## now rewards: batch_size * max_len
        BLEU = np.transpose(np.array(BLEU))

        if  bias_num is None:
            rewards = rewards * y_sample_mask  
            rewards = rewards / (1. * rollnum)
        else:
            bias = np.zeros_like(rewards)
            bias +=bias_num * rollnum
            rewards_minus_bias = rewards-bias

            rewards_minus_bias = namana * rewards_minus_bias + (1 - namana) * BLEU

            rewards=rewards_minus_bias * y_sample_mask
            rewards = rewards / (1. * rollnum)
        return rewards


    def init_and_restore(self, modelFile=None):
        params = tf.trainable_variables()
        init_op = tf.global_variables_initializer()
        saver = tf.train.Saver(params)

        self.sess.run(init_op)
        self.saver = saver
        if modelFile is None:
            self.saver.restore(self.sess, tf.train.latest_checkpoint(self.config.train.logdir))
        else:
            self.saver.restore(self.sess, modelFile)

    def build_test_model(self):
        """Build model for testing."""

        self.prepare(is_training=False)

        with self.graph.as_default():
            with tf.device(self.sync_device):
                self.src_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='src_pl')
                self.dst_pl = tf.placeholder(dtype=INT_TYPE, shape=[None, None], name='dst_pl')
                self.decoder_input = shift_right(self.dst_pl)
                Xs = split_tensor(self.src_pl, len(self.devices))
                Ys = split_tensor(self.dst_pl, len(self.devices))
                dec_inputs = split_tensor(self.decoder_input, len(self.devices))

                # Encode
                encoder_output_list = []
                for i, (X, device) in enumerate(zip(Xs, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        encoder_output = self.encoder(X, reuse=i > 0 or None)
                        encoder_output_list.append(encoder_output)
                self.encoder_output = tf.concat(encoder_output_list, axis=0)

                # Decode
                enc_outputs = split_tensor(self.encoder_output, len(self.devices))
                preds_list, k_preds_list, k_scores_list = [], [], []
                self.loss_sum = 0.0
                for i, (X, enc_output, dec_input, Y, device) in enumerate(zip(Xs, enc_outputs, dec_inputs, Ys, self.devices)):
                    with tf.device(lambda op: self.choose_device(op, device)):
                        self._logger.info('Build model on %s.' % device)
                        decoder_output = self.decoder(dec_input, enc_output, reuse=i > 0 or None)
                        # Predictions
                        preds, k_preds, k_scores = self.test_output(decoder_output, reuse=i > 0 or None)
                        preds_list.append(preds)
                        k_preds_list.append(k_preds)
                        k_scores_list.append(k_scores)
                        # Loss
                        loss = self.test_loss(decoder_output, Y, reuse=True)
                        self.loss_sum += loss

                self.preds = tf.concat(preds_list, axis=0)
                self.k_preds = tf.concat(k_preds_list, axis=0)
                self.k_scores = tf.concat(k_scores_list, axis=0)

    def choose_device(self, op, device):
        """Choose a device according the op's type."""
        if op.type.startswith('Variable'):
            return self.sync_device
        return device

    def encoder(self, encoder_input, reuse):
        encoder_padding = tf.equal(encoder_input, 0)
        
        encoder_output = bottom(encoder_input,
                                   vocab_size=self.config.src_vocab_size,
                                   dense_size=self.config.hidden_units,
                                   shared_embedding=self.config.train.shared_embedding,
                                   reuse=reuse,
                                   multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0)
        """Transformer encoder."""
        with tf.variable_scope("encoder", initializer=self._initializer, reuse=reuse):
            # Mask
           # encoder_padding = tf.equal(encoder_input, 0)
           # # Embedding
           # encoder_output = embedding(encoder_input,
           #                            vocab_size=self.config.src_vocab_size,
           #                            dense_size=self.config.hidden_units,
           #                            multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0,
           #                            name="src_embedding")
            # Add positional signal
            encoder_output = add_timing_signal_1d(encoder_output)
            # Dropout
            encoder_output = tf.layers.dropout(encoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)

            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention
                    encoder_output = residual(encoder_output,
                                              multihead_attention(
                                                  query_antecedent=encoder_output,
                                                  memory_antecedent=None,
                                                  bias=attention_bias_ignore_padding(encoder_padding),
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name='encoder_self_attention',
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    encoder_output = residual(encoder_output,
                                              conv_hidden_relu(
                                                  inputs=encoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

        # Mask padding part to zeros.
        encoder_output *= tf.expand_dims(1.0 - tf.to_float(encoder_padding), axis=-1)
        return encoder_output

    def decoder(self, decoder_input, encoder_output, reuse):
        encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
        encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)
        decoder_output = target(decoder_input,
                                   vocab_size=self.config.dst_vocab_size,
                                   dense_size=self.config.hidden_units,
                                   shared_embedding=self.config.train.shared_embedding,
                                   reuse=reuse,
                                   multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0)
        """Transformer decoder"""
        with tf.variable_scope("decoder", initializer=self._initializer, reuse=reuse):
            #encoder_padding = tf.equal(tf.reduce_sum(tf.abs(encoder_output), axis=-1), 0.0)
            #encoder_attention_bias = attention_bias_ignore_padding(encoder_padding)

            #decoder_output = embedding(decoder_input,
            #                           vocab_size=self.config.dst_vocab_size,
            #                           dense_size=self.config.hidden_units,
            #                           multiplier=self.config.hidden_units**0.5 if self.config.scale_embedding else 1.0,
            #                           name="dst_embedding")
            # Positional Encoding
            decoder_output += add_timing_signal_1d(decoder_output)
            # Dropout
            decoder_output = tf.layers.dropout(decoder_output,
                                               rate=self.config.residual_dropout_rate,
                                               training=self.is_training)
            # Bias for preventing peeping later information
            self_attention_bias = attention_bias_lower_triangle(tf.shape(decoder_input)[1])

            # Blocks
            for i in range(self.config.num_blocks):
                with tf.variable_scope("block_{}".format(i)):
                    # Multihead Attention (self-attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query_antecedent=decoder_output,
                                                  memory_antecedent=None,
                                                  bias=self_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  output_depth=self.config.hidden_units,
                                                  name="decoder_self_attention",
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Multihead Attention (vanilla attention)
                    decoder_output = residual(decoder_output,
                                              multihead_attention(
                                                  query_antecedent=decoder_output,
                                                  memory_antecedent=encoder_output,
                                                  bias=encoder_attention_bias,
                                                  total_key_depth=self.config.hidden_units,
                                                  total_value_depth=self.config.hidden_units,
                                                  output_depth=self.config.hidden_units,
                                                  num_heads=self.config.num_heads,
                                                  dropout_rate=self.config.attention_dropout_rate if self.is_training else 0.0,
                                                  name="decoder_vanilla_attention",
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

                    # Feed Forward
                    decoder_output = residual(decoder_output,
                                              conv_hidden_relu(
                                                  decoder_output,
                                                  hidden_size=4 * self.config.hidden_units,
                                                  output_size=self.config.hidden_units,
                                                  summaries=self._summary),
                                              dropout_rate=self.config.residual_dropout_rate,
                                              is_training=self.is_training)

            return decoder_output

    def test_output(self, decoder_output, reuse):
        last_logits = top(body_output=decoder_output[:, -1],
                     vocab_size = self.config.dst_vocab_size,
                     dense_size = self.config.hidden_units,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)
        """During test, we only need the last prediction."""
        with tf.variable_scope("output",initializer=self._initializer,  reuse=reuse):
            #last_logits = tf.layers.dense(decoder_output[:,-1], self.config.dst_vocab_size)
            last_preds = tf.to_int32(tf.arg_max(last_logits, dimension=-1))
            z = tf.nn.log_softmax(last_logits)
            last_k_scores, last_k_preds = tf.nn.top_k(z, k=self.config.test.beam_size, sorted=False)
            last_k_preds = tf.to_int32(last_k_preds)
        return last_preds, last_k_preds, last_k_scores

    def test_loss(self, decoder_output, Y, reuse):
        logits = top(body_output=decoder_output,
                     vocab_size = self.config.dst_vocab_size,
                     dense_size = self.config.hidden_units,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            #logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            mask = tf.to_float(tf.not_equal(Y, 0))
            labels = tf.one_hot(Y, depth=self.config.dst_vocab_size)
            loss = tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels)
            loss_sum = tf.reduce_sum(loss * mask)
        return loss_sum

    def gan_output(self, decoder_output, Y, reward, reuse):
        logits = top(body_output=decoder_output,
                     vocab_size = self.config.dst_vocab_size,
                     dense_size = self.config.hidden_units,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            #logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            l_shape=tf.shape(logits)
            probs = tf.nn.softmax(tf.reshape(logits, [-1, self.config.dst_vocab_size]))
            probs = tf.reshape(probs, [l_shape[0], l_shape[1], l_shape[2]])
            sample = tf.to_float(l_shape[0])
            #n_sample = l_shape[0] * tf.convert_to_tensor(1.0, dtype=tf.float32)
            g_loss = -tf.reduce_sum(
                tf.reduce_sum(tf.one_hot(tf.reshape(Y, [-1]), self.config.dst_vocab_size, 1.0, 0.0) *
                              tf.reshape(probs, [-1, self.config.dst_vocab_size]), 1) *
                              tf.reshape(reward, [-1]), 0) / sample
        return g_loss

    def train_output(self, decoder_output, Y, reuse):
        """Calculate loss and accuracy."""
        logits = top(body_output=decoder_output,
                     vocab_size = self.config.dst_vocab_size,
                     dense_size = self.config.hidden_units,
                     shared_embedding = self.config.train.shared_embedding,
                     reuse=reuse)
        with tf.variable_scope("output", initializer=self._initializer, reuse=reuse):
            #logits = tf.layers.dense(decoder_output, self.config.dst_vocab_size)
            preds = tf.to_int32(tf.arg_max(logits, dimension=-1))
            mask = tf.to_float(tf.not_equal(Y, 0))
            acc = tf.reduce_sum(tf.to_float(tf.equal(preds, Y)) * mask) / tf.reduce_sum(mask)

            # Smoothed loss
            loss = smoothing_cross_entropy(logits=logits, labels=Y, vocab_size=self.config.dst_vocab_size,
                                           confidence=1-self.config.train.label_smoothing)
            mean_loss = tf.reduce_sum(loss * mask) / (tf.reduce_sum(mask))

        return acc, mean_loss


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)
        else:
            # Average over the 'tower' dimension.
            grad = tf.concat(axis=0, values=grads)
            grad = tf.reduce_mean(grad, 0)

            # Keep in mind that the Variables are redundant because they are shared
            # across towers. So .. we will just return the first tower's pointer to
            # the Variable.
            v = grad_and_vars[0][1]
            grad_and_var = (grad, v)
            average_grads.append(grad_and_var)
    return average_grads


def residual(inputs, outputs, dropout_rate, is_training):
    """Residual connection.

    Args:
        inputs: A Tensor.
        outputs: A Tensor.
        dropout_rate: A float.
        is_training: A bool.

    Returns:
        A Tensor.
    """
    output = inputs + tf.layers.dropout(outputs, rate=dropout_rate, training=is_training)
    output = layer_norm(output)
    return output


def split_tensor(input, n):
    """
    Split the tensor input to n tensors.
    Args:
        inputs: A tensor with size [b, ...].
        n: A integer.

    Returns: A tensor list, each tensor has size [b/n, ...].
    """
    batch_size = tf.shape(input)[0]
    ls = tf.cast(tf.lin_space(0.0, tf.cast(batch_size, FLOAT_TYPE), n + 1), INT_TYPE)
    return [input[ls[i]:ls[i+1]] for i in range(n)]


def learning_rate_decay(config, global_step):
    """Inverse-decay learning rate until warmup_steps, then decay."""
    warmup_steps = tf.to_float(config.train.learning_rate_warmup_steps)
    global_step = tf.to_float(global_step)
    return config.hidden_units ** -0.5 * tf.minimum(
        (global_step + 1.0) * warmup_steps ** -1.5, (global_step + 1.0) ** -0.5)


def shift_right(input, pad=2):
    """Shift input tensor right to create decoder input. '2' denotes <S>"""
    return tf.concat((tf.ones_like(input[:, :1]) * pad, input[:, :-1]), 1)

def get_weight(vocab_size, dense_size, name=None):
     weights = tf.get_variable("kernel", [vocab_size, dense_size], initializer=tf.random_normal_initializer(0.0, 512**-0.5))
     return weights

def bottom(x, vocab_size, dense_size, shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope("embedding", reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=None):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("src_embedding", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def target(x, vocab_size, dense_size, shared_embedding=True, reuse=None, multiplier=1.0):
    with tf.variable_scope("embedding", reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
               embedding_var = get_weight(vocab_size, dense_size)
               emb_x = tf.gather(embedding_var, x)
               if multiplier != 1.0:
                   emb_x *= multiplier
        else:
            with tf.variable_scope("dst_embedding", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                emb_x = tf.gather(embedding_var, x)
                if multiplier !=1.0:
                    emb_x *= multiplier
    return emb_x

def top(body_output, vocab_size, dense_size, shared_embedding=True, reuse=None):
    with tf.variable_scope('embedding', reuse=reuse):
        if shared_embedding:
            with tf.variable_scope("shared", reuse=True):
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size])
                embedding_var = get_weight(vocab_size, dense_size)
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
        else:
            with tf.variable_scope("softmax", reuse=None):
                embedding_var = get_weight(vocab_size, dense_size)
                shape=tf.shape(body_output)[:-1]
                body_output = tf.reshape(body_output, [-1, dense_size]) 
                logits = tf.matmul(body_output, embedding_var, transpose_b=True)
                logits = tf.reshape(logits, tf.concat([shape, [vocab_size]], 0))
    return logits

def embedding(x, vocab_size, dense_size, name=None, reuse=None, multiplier=1.0):
    """Embed x of type int64 into dense vectors."""
    with tf.variable_scope(
        name, default_name="embedding", values=[x], reuse=reuse):
        embedding_var = tf.get_variable("kernel", [vocab_size, dense_size])
        emb_x = tf.gather(embedding_var, x)
        if multiplier != 1.0:
            emb_x *= multiplier
        return emb_x
