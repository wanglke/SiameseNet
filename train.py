import tensorflow as tf
import numpy as np
import re
import time
import datetime
import gc
from input_helpers import InputHelper
from siamese_network import SiameseLSTM
from siamese_network_semantic import SiameseLSTMw2v
from hyperparameters import Hyperparamters as hp
from tensorflow.contrib import learn
import gzip
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import warnings
warnings.filterwarnings("ignore")
from random import random
import os


if hp.training_files==None:
    print("Input Files List is empty. use --training_files argument.")
    exit()


max_document_length = 15
inpH = InputHelper()
train_set, dev_set, vocab_processor, sum_no_of_batches = inpH.getDataSets(hp.training_files, max_document_length, hp.percent_dev, hp.batch_size, hp.is_char_based)
trainableEmbeddings = False

if hp.is_char_based == True:
    hp.pretrain_model = False
else:
    if hp.pretrain_model==None:
        trainableEmbeddings = True
        print("word2vec model path is empty.")
    else:
        inpH.loadW2V(hp.pretrain_model, hp.pretrain_model_format)

# Training
# ==================================================
print("starting graph def")
with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=hp.allow_soft_placement,
      log_device_placement=hp.log_device_placement)
    sess = tf.Session(config=session_conf)
    print("started session")
    with sess.as_default():
        if hp.is_char_based:
            siameseModel = SiameseLSTM(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=hp.embedding_dim,
                hidden_units=hp.hidden_units,
                l2_reg_lambda=hp.l2_reg_lambda,
                batch_size=hp.batch_size
            )
        else:
            siameseModel = SiameseLSTMw2v(
                sequence_length=max_document_length,
                vocab_size=len(vocab_processor.vocabulary_),
                embedding_size=hp.embedding_dim,
                hidden_units=hp.hidden_units,
                l2_reg_lambda=hp.l2_reg_lambda,
                batch_size=hp.batch_size,
                trainableEmbeddings=trainableEmbeddings
            )
        # Define Training procedure
        global_step = tf.Variable(0, name="global_step", trainable=False)
        optimizer = tf.train.AdamOptimizer(1e-3)
        print("initialized siameseModel object")
    
    grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
    tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
    print("defined training_ops")
    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)
    print("defined gradient summaries")
    # Output directory for models and summaries
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "model_saved", hp.model_saved_dir))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", siameseModel.loss)
    acc_summary = tf.summary.scalar("accuracy", siameseModel.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph)

    # Dev summaries
    dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
    dev_summary_dir = os.path.join(out_dir, "summaries", "dev")
    dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, sess.graph)

    # Checkpoint directory. Tensorflow assumes this directory already exists so we need to create it
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

    # Write vocabulary
    vocab_processor.save(os.path.join(checkpoint_dir, "vocab"))

    # Initialize all variables
    sess.run(tf.global_variables_initializer())
    
    print("init all variables")
    graph_def = tf.get_default_graph().as_graph_def()
    graphpb_txt = str(graph_def)
    with open(os.path.join(checkpoint_dir, "graphpb.txt"), 'w') as f:
        f.write(graphpb_txt)

    if hp.pretrain_model:
        # initial matrix with random uniform
        initW = np.random.uniform(-0.25,0.25,(len(vocab_processor.vocabulary_), hp.embedding_dim))
        #initW = np.zeros(shape=(len(vocab_processor.vocabulary_), hp.embedding_dim))
        # load any vectors from the word2vec
        print("initializing initW with pre-trained word2vec embeddings")
        for w in vocab_processor.vocabulary_._mapping:
            arr=[]
            s = re.sub('[^0-9a-zA-Z]+', '', w)
            if w in inpH.pre_emb:
                arr=inpH.pre_emb[w]
            elif w.lower() in inpH.pre_emb:
                arr=inpH.pre_emb[w.lower()]
            elif s in inpH.pre_emb:
                arr=inpH.pre_emb[s]
            elif s.isdigit():
                arr=inpH.pre_emb["zero"]
            if len(arr)>0:
                idx = vocab_processor.vocabulary_.get(w)
                initW[idx]=np.asarray(arr).astype(np.float32)
        print("Done assigning intiW. len="+str(len(initW)))
        inpH.deletePreEmb()
        gc.collect()
        sess.run(siameseModel.W.assign(initW))

    def train_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: hp.dropout_keep_prob,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: hp.dropout_keep_prob,
            }
        _, step, loss, accuracy, dist, sim, summaries = sess.run([tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim, train_summary_op],  feed_dict)
        time_str = datetime.datetime.now().isoformat()
        if step % hp.print_step == 0:
            print("TRAIN {}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)
        # print(y_batch, dist, sim)

    def dev_step(x1_batch, x2_batch, y_batch):
        """
        A single training step
        """ 
        if random()>0.5:
            feed_dict = {
                siameseModel.input_x1: x1_batch,
                siameseModel.input_x2: x2_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        else:
            feed_dict = {
                siameseModel.input_x1: x2_batch,
                siameseModel.input_x2: x1_batch,
                siameseModel.input_y: y_batch,
                siameseModel.dropout_keep_prob: 1.0,
            }
        step, loss, accuracy, sim, summaries = sess.run([global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim, dev_summary_op],  feed_dict)

        dev_summary_writer.add_summary(summaries, step)
        # print (y_batch, sim)
        return step,loss,accuracy

    # Generate batches
    batches = inpH.batch_iter(list(zip(train_set[0], train_set[1], train_set[2])), hp.batch_size, hp.num_epochs)

    ptr=0
    max_validation_acc=0.0
    for nn in range(sum_no_of_batches*hp.num_epochs):
        batch = batches.__next__()
        if len(batch) < 1:
            continue
        x1_batch,x2_batch, y_batch = zip(*batch)
        if len(y_batch)<1:
            continue
        train_step(x1_batch, x2_batch, y_batch)
        current_step = tf.train.global_step(sess, global_step)
        sum_acc = 0.0
        sum_loss=0.0
        if current_step % hp.evaluate_every == 0:
            dev_batches = inpH.batch_iter(list(zip(dev_set[0], dev_set[1], dev_set[2])), hp.batch_size, 1)
            batch_num = 0
            for db in dev_batches:
                if len(db)<1:
                    continue
                x1_dev_b,x2_dev_b,y_dev_b = zip(*db)
                if len(y_dev_b)<1:
                    continue
                step, loss, acc = dev_step(x1_dev_b, x2_dev_b, y_dev_b)
                sum_acc = sum_acc + acc
                sum_loss = sum_loss + loss
                batch_num += 1
            time_str = datetime.datetime.now().isoformat()
            print("\nDEV {}: step {}, loss {:g}, acc {:g}".format(time_str, step, sum_loss/batch_num, sum_acc/batch_num))
        if current_step % hp.checkpoint_every == 0:
            if sum_acc >= max_validation_acc:
                max_validation_acc = sum_acc
                saver.save(sess, checkpoint_prefix, global_step=current_step)
                tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, "graph"+str(nn)+".pb", as_text=False)
                print("Saved model {} with sum_accuracy={} checkpoint to {}\n".format(nn, max_validation_acc, checkpoint_prefix))
