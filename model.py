# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
# Modifications Copyright 2017 Abigail See
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""This file contains code to build and run the tensorflow graph for the sequence-to-sequence model"""

import os
import time
import numpy as np
import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import config

FLAGS = tf.app.flags.FLAGS



class DiscriminatorModel(object):
  """A class to represent a sequence-to-sequence model for text summarization. Supports both baseline mode, pointer-generator mode, and coverage"""

  def __init__(self, mode, vocab):
    self._mode = mode
    self._vocab = vocab

  def _add_placeholders(self):
    """Add placeholders to the graph. These are entry points for any input data."""


    # encoder part
    self._enc_batch = tf.placeholder(tf.int32, [config.batch_size, None], name='enc_batch')
    self._enc_lens = tf.placeholder(tf.int32, [config.batch_size], name='enc_lens')
    self._enc_padding_mask = tf.placeholder(tf.float32, [config.batch_size, None], name='enc_padding_mask')


    # decoder part
    self._dec_batch = tf.placeholder(tf.int32, [config.batch_size, None], name='dec_batch')
    self._dec_lens = tf.placeholder(tf.int32, [config.batch_size], name='dec_lens')
    self._dec_padding_mask = tf.placeholder(tf.float32, [config.batch_size, None], name='dec_padding_mask')

    #targets
    self._targets = tf.placeholder(tf.float32, [config.batch_size, 2], name='target')

  def build_graph(self):
    """Add the placeholders, model, global step, train_op and summaries to the graph"""
    tf.logging.info('Building graph...')
    t0 = time.time()

    self._add_placeholders()

    with tf.device("/gpu:%d"%(config.gpu_selection)):
        self._add_seq2seq()

    self.global_step = tf.Variable(0, name='global_step', trainable=False)


    if self._mode == 'train':
      self._add_train_op()

    self._summaries = tf.summary.merge_all()

    t1 = time.time()
    tf.logging.info('Time to build graph: %i seconds', t1 - t0)

  def _add_emb_vis(self, embedding_var):
    """Do setup so that we can view word embedding visualization in Tensorboard, as described here:
    https://www.tensorflow.org/get_started/embedding_viz
    Make the vocab metadata file, then make the projector config file pointing to it."""
    train_dir = os.path.join(FLAGS.log_root, "train")
    vocab_metadata_path = os.path.join(train_dir, "vocab_metadata.tsv")
    self._vocab.write_metadata(vocab_metadata_path) # write metadata file
    summary_writer = tf.summary.FileWriter(train_dir)
    projconfig = projector.ProjectorConfig()
    embedding = projconfig.embeddings.add()
    embedding.tensor_name = embedding_var.name
    embedding.metadata_path = vocab_metadata_path
    projector.visualize_embeddings(summary_writer, projconfig)

  def _add_input_encoder(self, inputs, seq_len):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope("encoder"):
      cell_fw = tf.contrib.rnn.LSTMCell(config.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(config.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      ((fw_states, bw_states), (final_fw, final_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True)

    return fw_states, bw_states, final_fw, final_bw

  def _add_input_decoder(self, inputs, seq_len, enc_fw, enc_bw):
    """Add a single-layer bidirectional LSTM encoder to the graph.

    Args:
      encoder_inputs: A tensor of shape [batch_size, <=max_enc_steps, emb_size].
      seq_len: Lengths of encoder_inputs (before padding). A tensor of shape [batch_size].

    Returns:
      outputs:
        A tensor of shape [batch_size, <=max_enc_steps, 2*hidden_dim]. It's 2*hidden_dim because it's the concatenation of the forwards and backwards states.
      fw_state, bw_state:
        Each are LSTMStateTuples of shape ([batch_size,hidden_dim],[batch_size,hidden_dim])
    """
    with tf.variable_scope("decoder"):
      cell_fw = tf.contrib.rnn.LSTMCell(config.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      cell_bw = tf.contrib.rnn.LSTMCell(config.hidden_dim, initializer=self.rand_unif_init, state_is_tuple=True)
      ((fw_states, bw_states), (final_fw, final_bw)) = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, dtype=tf.float32, sequence_length=seq_len, swap_memory=True, initial_state_fw=enc_fw, initial_state_bw=enc_bw)

    return fw_states, bw_states



  def _add_seq2seq(self):
    """Add the whole sequence-to-sequence model to the graph."""
    mode = self._mode
    vsize = self._vocab.size() # size of the vocabulary

    with tf.variable_scope('seq2seq'):
      # Some initializers
      self.rand_unif_init = tf.random_uniform_initializer(-config.rand_unif_init_mag, config.rand_unif_init_mag, seed=123)
      self.trunc_norm_init = tf.truncated_normal_initializer(stddev=config.trunc_norm_init_std)

      # Add embedding matrix (shared by the encoder and decoder inputs)
      with tf.variable_scope('embedding'):
        embedding = tf.get_variable('embedding', [vsize, config.emb_dim], dtype=tf.float32, initializer=self.trunc_norm_init)
        if mode=="train": self._add_emb_vis(embedding) # add to tensorboard
        emb_enc_inputs = tf.nn.embedding_lookup(embedding, self._enc_batch) # tensor with shape (batch_size, max_enc_steps, emb_size)
        emb_dec_inputs = tf.nn.embedding_lookup(embedding, self._dec_batch) # tensor with shape (batch_size, max_dec_steps, emb_size)
        #emb_dec_inputs = [tf.nn.embedding_lookup(embedding, x) for x in tf.unstack(self._dec_batch, axis=1)] # list length max_dec_steps containing shape (batch_size, emb_size)

      # Add the encoder.
      enc_fw_states, enc_bw_states, enc_fw, enc_bw = self._add_input_encoder(emb_enc_inputs, self._enc_lens)

      print("Encoder FW", enc_fw_states.shape)
      print("Encoder BW", enc_bw_states.shape)
      raise Exception("testing mode")

      #reshape encoder states from [batch_size, input_size, hidden_dim] to [batch_size, input_size * hidden_dim]
      enc_fw_states = tf.reshape(enc_fw_states, [config.batch_size, config.hidden_dim * tf.shape(enc_fw_states)[1]])
      enc_bw_states = tf.reshape(enc_bw_states, [config.batch_size, config.hidden_dim * tf.shape(enc_bw_states)[1]])


      # python run.py --mode=decode --data_path=data/chunked/train_1/train_1_*.bin --vocab_path=data/vocab_1 --exp_name=full1isto1

      # Add the decoder.
      dec_fw_states, dec_bw_states = self._add_input_decoder(emb_dec_inputs, self._dec_lens, enc_fw, enc_bw)

      #reshape decoder states from [batch_size, input_size, hidden_dim] to [batch_size, input_size * hidden_dim]
      dec_fw_states = tf.reshape(dec_fw_states, [config.batch_size, config.hidden_dim * tf.shape(dec_fw_states)[1]])
      dec_bw_states = tf.reshape(dec_bw_states, [config.batch_size, config.hidden_dim * tf.shape(dec_bw_states)[1]])
      #print("Decoder FW", dec_fw_states.shape)
      #print("Decoder BW", dec_bw_states.shape)


      #enc_c = tf.concat(axis=1, values=[enc_fw.c, enc_bw.c])
      #enc_h = tf.concat(axis=1, values=[enc_fw.h, enc_bw.h])
      #dec_c = tf.concat(axis=1, values=[dec_fw.c, dec_bw.c])
      #dec_h = tf.concat(axis=1, values=[dec_fw.h, dec_bw.h])

      final_encoding = tf.concat(axis=1, values=[enc_fw_states, enc_bw_states, dec_fw_states, dec_bw_states])
      #print("Final encoding", final_encoding.shape)
      #raise Exception("Test")
      dims_final_enc = tf.shape(final_encoding)

      """
      #convo_input = tf.concat(axis=1, values=[enc_c, enc_h, dec_c, dec_h])
      input_layer = tf.reshape(final_encoding, [config.batch_size, dims_final_enc[1], 1])
      print("Convolution input shape", input_layer.shape)

      conv1 = tf.layers.conv1d(
      inputs=input_layer,
      filters=8,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)
      conv1 = tf.layers.batch_normalization(conv1)
      print("Convolution1 output shape", conv1.shape)

      pool1 = tf.layers.max_pooling1d(inputs=conv1, pool_size=2, strides=2)
      print("Pool1 output shape", pool1.shape)

      conv2 = tf.layers.conv1d(
      inputs=pool1,
      filters=16,
      kernel_size=5,
      padding="same",
      activation=tf.nn.relu)


      conv2 = tf.layers.batch_normalization(conv2)
      print("Convolution2 output shape", conv2.shape)

      pool2 = tf.layers.max_pooling1d(inputs=conv2, pool_size=2, strides=2)
      print("Pool2 output shape", pool2.shape)

      dims_pool2 = tf.shape(pool2)

      pool2_flat = tf.reshape(pool2, [config.batch_size, dims_pool2[1] * 16])
      print("Pool2_flat output shape", pool2_flat.shape)
      dense = tf.layers.dense(inputs=pool2_flat, units=1024, activation=tf.nn.relu)
      """
      #raise Exception("testing mode")

      #dropout = tf.layers.dropout(inputs=dense, rate=0.4, training=mode=="train")
      #print("Dense output shape", dense.shape)

      #raise Exception("Just testing")
      # Add the output projection to obtain the vocabulary distribution
      with tf.variable_scope('output_projection'):
        w = tf.get_variable('w', [dims_final_enc[1], 2], dtype=tf.float32, initializer=self.trunc_norm_init)
        bias_output = tf.get_variable('bias_output', [2], dtype=tf.float32, initializer=self.trunc_norm_init)
        #concatenate abstract and article outputs [batch_size, hidden_dim*4]


        #get classification output [batch_size, 1] default on last axis
        self._logits = tf.matmul(final_encoding, w) + bias_output
        #self._logits = tf.layers.dense(final_encoding, 2, kernel_initializer=self.trunc_norm_init, bias_initializer=self.trunc_norm_init)
        #self._prob = tf.nn.softmax(logits, "class_prob")

      if mode in ['train', 'eval']:
        # Calculate the loss
        with tf.variable_scope('loss'):
          #self._prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=self._targets)
          #class_weights = tf.constant([0.1, 5.])
          self._loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self._targets, logits=self._logits))
          #self._loss = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(targets=self._targets, logits=self._logits, pos_weight=class_weights))
          tf.summary.scalar('loss', self._loss)



    #if mode == "decode":

  def _add_train_op(self):
    """Sets self._train_op, the op to run for training."""
    # Take gradients of the trainable variables w.r.t. the loss function to minimize
    loss_to_minimize = self._loss
    tvars = tf.trainable_variables()
    gradients = tf.gradients(loss_to_minimize, tvars, aggregation_method=tf.AggregationMethod.EXPERIMENTAL_TREE)

    # Clip the gradients
    with tf.device("/gpu:%d"%(config.gpu_selection)):
      grads, global_norm = tf.clip_by_global_norm(gradients, config.max_grad_norm)

    # Add a summary
    tf.summary.scalar('global_norm', global_norm)

    # Apply adagrad optimizer
    optimizer = tf.train.AdagradOptimizer(config.lr, initial_accumulator_value=config.adagrad_init_acc)
    #optimizer = tf.train.MomentumOptimizer(config.lr, momentum=0.01)
    with tf.device("/gpu:%d"%(config.gpu_selection)):
      self._train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step, name='train_step')


  def run_train_step(self, sess, batch):
    """Runs one training iteration. Returns a dictionary containing train op, summaries, loss, global_step"""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'train_op': self._train_op,
        'summaries': self._summaries,
        'loss': self._loss,
        'logits': self._logits,
        'global_step': self.global_step,
    }

    return sess.run(to_return, feed_dict)

  def run_eval_step(self, sess, batch):
    """Runs one evaluation iteration. Returns a dictionary containing summaries, loss, global_step and (optionally) coverage loss."""
    feed_dict = self._make_feed_dict(batch)
    to_return = {
        'summaries': self._summaries,
        'loss': self._loss,
        'logits': self._logits,
        'global_step': self.global_step,
    }

    return sess.run(to_return, feed_dict)

  def run_decode_step(self, sess, batch):
    feed_dict = self._make_feed_dict(batch, just_enc=True)
    to_return = {
        'logits': self._logits,
    }

    return sess.run(to_return, feed_dict)

  def _make_feed_dict(self, batch, just_enc=False):
    """Make a feed dictionary mapping parts of the batch to the appropriate placeholders.

    Args:
      batch: Batch object
      just_enc: Boolean. If True, only feed the parts needed for the encoder.
    """
    feed_dict = {}
    feed_dict[self._enc_batch] = batch.enc_batch
    feed_dict[self._enc_lens] = batch.enc_lens
    feed_dict[self._enc_padding_mask] = batch.enc_padding_mask
    feed_dict[self._dec_batch] = batch.dec_batch
    feed_dict[self._dec_lens] = batch.dec_lens
    feed_dict[self._dec_padding_mask] = batch.dec_padding_mask


    if not just_enc:

      feed_dict[self._targets] = batch.targets


    return feed_dict
