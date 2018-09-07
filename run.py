import tensorflow as tf
import numpy as np
from data import Vocab
import os
import config
from batcher import Batcher
from model_attention import DiscriminatorModel
import util
import time
import pickle as pkl
FLAGS = tf.app.flags.FLAGS

#os.nice(4)

#python run.py --mode=train --data_path=data/chunked/train_tout/train_tout_*.bin --vocab_path=data/vocab_tout --exp_name=attnmodel_weighted_tout --sel_gpu=0 --weighted_model=1
#python run.py --mode=eval --data_path=data/chunked/val_tout/val_tout_*.bin --vocab_path=data/vocab_tout --exp_name=attnmodel_weighted_tout --sel_gpu=1 --weighted_model=1

#python run.py --mode=train --data_path=data/chunked/train_1/train_1_*.bin --vocab_path=data/vocab_1 --exp_name=attn1isto1 --sel_gpu=0
#python run.py --mode=eval --data_path=data/chunked/val_1/val_1_*.bin --vocab_path=data/vocab_1 --exp_name=attn1isto1 --sel_gpu=1
#python run.py --mode=decode --data_path=data/chunked/test_1/test_1_*.bin --vocab_path=data/vocab_1 --exp_name=attn1isto1 --sel_gpu=1 --single_pass=1

#START defining flags
tf.app.flags.DEFINE_boolean('sel_gpu', 0, 'GPU device number')
# Where to find data
tf.app.flags.DEFINE_string('data_path', "", 'Path expression to tf.Example datafiles. Can include wildcards to access multiple datafiles.')
tf.app.flags.DEFINE_string('vocab_path', "", 'Path expression to text vocabulary file.')

# Important settings
tf.app.flags.DEFINE_string('mode', '', 'must be one of train/eval/decode')
tf.app.flags.DEFINE_string('log_root', '', 'Root directory for all logging.')
tf.app.flags.DEFINE_string('exp_name', '', 'Name for experiment. Logs will be saved in a directory with this name, under log_root.')
#can be overwritten from command line
tf.app.flags.DEFINE_boolean('single_pass', config.single_pass, 'For decode mode only. If True, run eval on the full dataset using a fixed checkpoint, i.e. take the current checkpoint, and use it to produce one summary for each example in the dataset, write the summaries to file and then get ROUGE scores for the whole dataset. If False (default), run concurrent decoding, i.e. repeatedly load latest checkpoint, use it to produce summaries for randomly-chosen examples and log the results to screen, indefinitely.')

# Coverage hyperparameters
#can be overwritten from command line
tf.app.flags.DEFINE_boolean('coverage', config.coverage, 'Use coverage mechanism. Note, the experiments reported in the ACL paper train WITHOUT coverage until converged, and then train for a short phase WITH coverage afterwards. i.e. to reproduce the results in the ACL paper, turn this off for most of training then turn on for a short phase at the end.')

# Utility flags, for restoring and changing checkpoints
#can be overwritten from command line
tf.app.flags.DEFINE_boolean('convert_to_coverage_model', config.convert_to_coverage_model, 'Convert a non-coverage model to a coverage model. Turn this on and run in train mode. Your current training model will be copied to a new version (same name with _cov_init appended) that will be ready to run with coverage flag turned on, for the coverage training stage.')
#can be overwritten from command line
tf.app.flags.DEFINE_boolean('restore_best_model', config.restore_best_model, 'Restore the best model in the eval/ dir and save it in the train/ dir, ready to be used for further training. Useful for early stopping, or if your training checkpoint has become corrupted with e.g. NaN values.')

tf.app.flags.DEFINE_boolean('weighted_loss', False, 'Use weighted loss for class immbalance')

# Debugging. See https://www.tensorflow.org/programmers_guide/debugger
tf.app.flags.DEFINE_boolean('debug', config.debug, "Run in tensorflow's debug mode (watches for NaN/inf values)")

def calc_running_avg_loss(loss, running_avg_loss, summary_writer, step, decay=0.99):
  """Calculate the running average loss via exponential decay.
  This is used to implement early stopping w.r.t. a more smooth loss curve than the raw loss curve.

  Args:
    loss: loss on the most recent eval step
    running_avg_loss: running_avg_loss so far
    summary_writer: FileWriter object to write for tensorboard
    step: training iteration step
    decay: rate of exponential decay, a float between 0 and 1. Larger is smoother.

  Returns:
    running_avg_loss: new running average loss
  """
  if running_avg_loss == 0:  # on the first iteration just take the loss
    running_avg_loss = loss
  else:
    running_avg_loss = running_avg_loss * decay + (1 - decay) * loss
  running_avg_loss = min(running_avg_loss, 12)  # clip
  loss_sum = tf.Summary()
  tag_name = 'running_avg_loss/decay=%f' % (decay)
  loss_sum.value.add(tag=tag_name, simple_value=running_avg_loss)
  summary_writer.add_summary(loss_sum, step)
  tf.logging.info('running_avg_loss: %f', running_avg_loss)
  return running_avg_loss


def restore_best_model():
  """Load bestmodel file from eval directory, add variables for adagrad, and save to train directory"""
  tf.logging.info("Restoring bestmodel for training...")

  # Initialize all vars in the model
  sess = tf.Session(config=util.get_config())
  print("Initializing all variables...")
  sess.run(tf.initialize_all_variables())

  # Restore the best model from eval dir
  saver = tf.train.Saver([v for v in tf.all_variables() if "Adagrad" not in v.name])
  print("Restoring all non-adagrad variables from best model in eval dir...")
  curr_ckpt = util.load_ckpt(saver, sess, "eval")
  print("Restored %s." % curr_ckpt)

  # Save this model to train dir and quit
  new_model_name = curr_ckpt.split("/")[-1].replace("bestmodel", "model")
  new_fname = os.path.join(FLAGS.log_root, "train", new_model_name)
  print("Saving model to %s..." % (new_fname))
  new_saver = tf.train.Saver() # this saver saves all variables that now exist, including Adagrad variables
  new_saver.save(sess, new_fname)
  print("Saved.")
  exit()

#start training
def setup_training(model, batcher):
  """Does setup before starting training (run_training)"""
  train_dir = os.path.join(FLAGS.log_root, "train")
  if not os.path.exists(train_dir): os.makedirs(train_dir)

  model.build_graph() # build the graph
  if FLAGS.restore_best_model:
    restore_best_model()

  saver = tf.train.Saver(max_to_keep=3) # keep 3 checkpoints at a time

  sv = tf.train.Supervisor(logdir=train_dir,
                     is_chief=True,
                     saver=saver,
                     summary_op=None,
                     save_summaries_secs=60, # save summaries for tensorboard every 60 secs
                     save_model_secs=60, # checkpoint every 60 secs
                     global_step=model.global_step)

  summary_writer = sv.summary_writer
  tf.logging.info("Preparing or waiting for session...")
  sess_context_manager = sv.prepare_or_wait_for_session(config=util.get_config())
  tf.logging.info("Created session.")

  try:
    run_training(model, batcher, sess_context_manager, sv, summary_writer) # this is an infinite loop until interrupted
  except KeyboardInterrupt:
    tf.logging.info("Caught keyboard interrupt on worker. Stopping supervisor...")
    sv.stop()

def run_training(model, batcher, sess_context_manager, sv, summary_writer):
  """Repeatedly runs training iterations, logging loss to screen and writing summaries"""
  tf.logging.info("starting run_training")
  with sess_context_manager as sess:
    if FLAGS.debug: # start the tensorflow debugger
      sess = tf_debug.LocalCLIDebugWrapperSession(sess)
      sess.add_tensor_filter("has_inf_or_nan", tf_debug.has_inf_or_nan)
    flag=0
    while True: # repeats until interrupted
      t0=time.time()
      #python3 run.py --data_path=data/chunked/val/val_*.bin --vocab_path=data/vocab --mode=eval --exp_name=modeltest
      batch = batcher.next_batch()
      while(batch is None and flag==0):
          batch = batcher.next_batch()
      flag = 1
      tf.logging.info('running training step...')

      results = model.run_train_step(sess, batch)
      t1=time.time()
      tf.logging.info('seconds for training step: %.3f', t1-t0)

      loss = results['loss']
      logits = results['logits']
      print(logits)
      targets = batch.targets
      predictions = np.argmax(logits, axis=1)
      targets = np.argmax(targets, axis=1)
      errors = np.count_nonzero(targets-predictions)
      #print(targets)
      tf.logging.info('loss: %f, errors: %d', loss, errors) # print the loss to screen

      if not np.isfinite(loss):
        raise Exception("Loss is not finite. Stopping.")

      # get the summaries and iteration number so we can write summaries to tensorboard
      summaries = results['summaries'] # we will write these summaries to tensorboard using summary_writer
      train_step = results['global_step'] # we need this to update our running average loss

      summary_writer.add_summary(summaries, train_step) # write the summaries
      if train_step % 100 == 0: # flush the summary writer every so often
        summary_writer.flush()


def run_eval(model, batcher, vocab):
  """Repeatedly runs eval iterations, logging to screen and writing summaries. Saves the model with the best loss seen so far."""
  model.build_graph() # build the graph
  saver = tf.train.Saver(max_to_keep=3) # we will keep 3 best checkpoints at a time
  sess = tf.Session(config=util.get_config())
  eval_dir = os.path.join(FLAGS.log_root, "eval") # make a subdir of the root dir for eval data
  bestmodel_save_path = os.path.join(eval_dir, 'bestmodel') # this is where checkpoints of best models are saved
  summary_writer = tf.summary.FileWriter(eval_dir)
  running_avg_loss = 0 # the eval job keeps a smoother, running average loss to tell it when to implement early stopping
  best_loss = None  # will hold the best loss achieved so far
  flag = 0
  while True:
    _ = util.load_ckpt(saver, sess) # load a new checkpoint
    batch = batcher.next_batch() # get the next batch
    while(batch is None and flag==0):
        batch = batcher.next_batch()
    flag = 1
    # run eval on the batch
    t0=time.time()
    results = model.run_eval_step(sess, batch)
    t1=time.time()
    tf.logging.info('seconds for batch: %.2f', t1-t0)

    # print the loss and coverage loss to screen
    loss = results['loss']
    logits = results['logits']
    targets = batch.targets
    predictions = np.argmax(logits, axis=1)
    targets = np.argmax(targets, axis=1)
    errors = np.count_nonzero(targets-predictions)
    tf.logging.info('loss: %f errors: %d', loss, errors)


    # add summaries
    summaries = results['summaries']
    train_step = results['global_step']
    summary_writer.add_summary(summaries, train_step)

    # calculate running avg loss
    running_avg_loss = calc_running_avg_loss(np.asscalar(loss), running_avg_loss, summary_writer, train_step)

    # If running_avg_loss is best so far, save this checkpoint (early stopping).
    # These checkpoints will appear as bestmodel-<iteration_number> in the eval dir
    if best_loss is None or running_avg_loss < best_loss:
      tf.logging.info('Found new best model with %.3f running_avg_loss. Saving to %s', running_avg_loss, bestmodel_save_path)
      saver.save(sess, bestmodel_save_path, global_step=train_step, latest_filename='checkpoint_best')
      best_loss = running_avg_loss

    # flush the summary writer every so often
    if train_step % 100 == 0:
      summary_writer.flush()

def get_decode_dir_name(ckpt_name):
  """Make a descriptive name for the decode dir, including the name of the checkpoint we use to decode. This is called in single_pass mode."""

  if "train" in FLAGS.data_path: dataset = "train"
  elif "val" in FLAGS.data_path: dataset = "val"
  elif "test" in FLAGS.data_path: dataset = "test"
  elif "ilp" in FLAGS.data_path: dataset = "ilp"
  elif "namas" in FLAGS.data_path: dataset = "namas"
  elif "seq2seq" in FLAGS.data_path: dataset = "seq2seq"
  elif "t3" in FLAGS.data_path: dataset = "t3"
  else: raise ValueError("FLAGS.data_path %s should contain one of train, val or test" % (FLAGS.data_path))
  dirname = "decode_%s" % (dataset)
  #if ckpt_name is not None
    #  dirname += "_%s" % ckpt_name
  return dirname

def run_decode(model, batcher, vocab):
    model.build_graph()
    saver = tf.train.Saver()
    sess = tf.Session(config=util.get_config())
    ckpt_path = util.load_ckpt(saver, sess)

    if FLAGS.single_pass:
        ckpt_name = "ckpt-" + ckpt_path.split('-')[-1] # this is something of the form "ckpt-123456"
        decode_dir = os.path.join(FLAGS.log_root, get_decode_dir_name(ckpt_name))
        #if os.path.exists(decode_dir):
        #    raise Exception("single_pass decode directory %s should not already exist" % decode_dir)

    else: # Generic decode dir name
        decode_dir = os.path.join(FLAGS.log_root, "decode")

    # Make the decode dir if necessary
    if not os.path.exists(decode_dir): os.mkdir(decode_dir)

    if FLAGS.single_pass:
      decode_prob_dir = os.path.join(decode_dir, "probs")
      if not os.path.exists(decode_prob_dir): os.mkdir(decode_prob_dir)
    flag = 0
    probs = []
    count=0
    CM = [[0, 0], [0, 0]]
    while True:
      batch = batcher.next_batch()
      while(batch is None and flag==0):
          batch = batcher.next_batch()
          flag = 1


      if (batch is None):
          assert FLAGS.single_pass, "Dataset exhausted, but we are not in single_pass mode"
          tf.logging.info("Decoder has finished reading dataset for single_pass.")
          break
      else:
          t0=time.time()
          results = model.run_decode_step(sess, batch)
          t1=time.time()
          #tf.logging.info('seconds for batch: %.2f', t1-t0)
          #choose any logit all are same
          logits = results['logits'][0]
          #print(logits)
          #define softmax probability for two classes
          x = logits[0]-logits[1]
          #find probability of smic
          p0 = 1. / (1. + np.exp(-x))
          probs.append(logits)
          targets = batch.targets[0]
          prediction = np.argmax(logits)
          target = np.argmax(targets)
          CM[prediction][target] += 1
          count+=1
          print("\rExample %d smic probability %f"%(count, p0),end="")
          print(CM)
          #print(targets)
    if FLAGS.single_pass:
        pkl.dump([probs, count],open(os.path.join(decode_prob_dir, 'probs.pkl'), 'wb'))
        print("\nProbability written in %s"% os.path.join(decode_prob_dir, 'probs.pkl'))


def main(unused_argv):
    tf.logging.set_verbosity(tf.logging.INFO) # choose what level of logging you want
    if(FLAGS.exp_name==''):
        raise Exception('You must set exp_name')
    FLAGS.log_root = os.path.join(FLAGS.log_root, FLAGS.exp_name)
    #create log dir in train mode else raise error
    if not os.path.exists(FLAGS.log_root):
      if FLAGS.mode.lower()=="train":
        os.makedirs(FLAGS.log_root)
      else:
        raise Exception("Logdir %s doesn't exist. Run in train mode to create it." % (FLAGS.log_root))

    # If single_pass=True, check we're in decode mode
    if FLAGS.single_pass and FLAGS.mode!='decode':
      raise Exception("The single_pass flag should only be True in decode mode")

    # create a vocabulary
    vocab = Vocab(FLAGS.vocab_path, config.vocab_size)

    # Create a batcher object that will create minibatches of data
    batcher = Batcher(FLAGS.data_path, vocab, FLAGS.mode, single_pass=FLAGS.single_pass)


    #ex = data.example_generator(FLAGS.data_path, FLAGS.single_pass).__next__()
    #abstract_text = ex.features.feature['abstract'].bytes_list.value[0].decode("utf-8")  # the abstract
    #print(abstract_text)
    #example = batcher._example_queue.get()
    #print(example.original_article)
    #print(example.original_abstract)
    #print(example.original_label)
    tf.set_random_seed(111) # a seed value for randomness

    if(FLAGS.mode.lower()=="train"):
        print("Starting training model")
        model = DiscriminatorModel(FLAGS.mode, vocab)
        setup_training(model, batcher)
    elif(FLAGS.mode.lower()=="eval"):
        print("Mode eval")
        model = DiscriminatorModel(FLAGS.mode, vocab)
        run_eval(model, batcher, vocab)
    elif(FLAGS.mode.lower()=="decode"):
        print("Mode decode")

        model = DiscriminatorModel(FLAGS.mode, vocab)
        run_decode(model, batcher, vocab)

    else:
        raise Exception("Mode must be one of train/eval/decode")

if __name__ == '__main__':

  tf.app.run()
