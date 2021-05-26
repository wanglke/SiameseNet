#! /usr/bin/env python
import tensorflow as tf
import numpy as np
import time
import datetime
from tensorflow.contrib import learn
from input_helpers import InputHelper
from hyperparameters import Hyperparamters as hp
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

def get_eval():
    if hp.eval_filepath == None or hp.vocab_filepath == None or hp.pretrain_model == None:
        print("Eval or Vocab filepaths are empty.")
        exit()

    trainableEmbeddings = False
    inpH = InputHelper()
    if hp.is_char_based == True:
        hp.pretrain_model = False
    else:
        if hp.pretrain_model == None:
            trainableEmbeddings = True
            print("word2vec model path is empty.")
        else:
            inpH.loadW2V(hp.pretrain_model, hp.pretrain_model_format)

    x1_test, x2_test, y_test = inpH.getTestDataSet(hp.eval_filepath, hp.vocab_filepath, 30)
    checkpoint_file = hp.checkpoint_file
    graph = tf.Graph()

    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=hp.allow_soft_placement,
                                      log_device_placement=hp.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            # Get the placeholders from the graph by name
            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            input_y = graph.get_operation_by_name("input_y").outputs[0]

            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            # Tensors we want to evaluate
            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

            # emb = graph.get_operation_by_name("embedding/W").outputs[0]
            # embedded_chars = tf.nn.embedding_lookup(emb,input_x)
            # Generate batches for one epoch
            batches = inpH.batch_iter(list(zip(x1_test, x2_test)), 2 * hp.batch_size, 1, shuffle=False)
            # Collect the predictions here
            predictions = []
            for db in batches:
                x1_dev_b, x2_dev_b = zip(*db)
                batch_sim = sess.run(sim, {input_x1: x1_dev_b, input_x2: x2_dev_b, dropout_keep_prob: 1.0})
                predictions = np.concatenate([predictions, batch_sim])

            ADD = predictions + y_test
            SUB = predictions - y_test
            TP = np.sum(ADD==2)
            TN = np.sum(ADD==0)
            FP = np.sum(SUB==1)
            FN = np.sum(SUB==-1)

            print(list(y_test).count(0))
            print(list(y_test).count(1))


            print("predictions")
            print(list(predictions).count(0))
            print(list(predictions).count(1))
            print(predictions)




            ACC = (TP+TN)/(TP+TN+FP+FN)
            R = TP/(TP+FN)
            P = TP/(TP+FP)
            F1= (2*(P*R))/(P+R)
            print("ACC:{}\nR:{}\nP:{}\nF1:{}".format(ACC, R , P,F1))




if __name__ == "__main__":
    get_eval()
