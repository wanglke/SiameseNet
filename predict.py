#! /usr/bin/env python
import tensorflow as tf
import numpy as np
from input_helpers import InputHelper
from hyperparameters import Hyperparamters as hp
from preprocess import cut_list_by_size

def get_sim_lable(sen_a, sen_b):
    if hp.eval_filepath==None or hp.vocab_filepath==None or hp.checkpoint_file==None :
        print("Eval or Vocab filepaths are empty.")
        exit()
    # load data and map id-transform based on training time vocabulary
    inpH = InputHelper()
    x1_test, x2_test = inpH.getTestData([sen_a], [sen_b], hp.vocab_filepath, 30)
    checkpoint_file = hp.checkpoint_file
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=hp.allow_soft_placement, log_device_placement = hp.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]
            sim = sess.run(sim, {input_x1: x1_test, input_x2: x2_test, dropout_keep_prob: 1.0})
        return int(sim[0])

def get_sim_lable_mutil(sen_a, sen_b):
    if hp.eval_filepath==None or hp.vocab_filepath==None or hp.checkpoint_file==None :
        print("Eval or Vocab filepaths are empty.")
        exit()
    # load data and map id-transform based on training time vocabulary
    inpH = InputHelper()
    x1_test, x2_test = inpH.getTestData(sen_a, sen_b, hp.vocab_filepath, 30)
    checkpoint_file = hp.checkpoint_file
    graph = tf.Graph()
    with graph.as_default():
        session_conf = tf.ConfigProto(allow_soft_placement=hp.allow_soft_placement, log_device_placement = hp.log_device_placement)
        sess = tf.Session(config=session_conf)
        with sess.as_default():
            # Load the saved meta graph and restore variables
            saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
            sess.run(tf.initialize_all_variables())
            saver.restore(sess, checkpoint_file)

            input_x1 = graph.get_operation_by_name("input_x1").outputs[0]
            input_x2 = graph.get_operation_by_name("input_x2").outputs[0]
            dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]
            sim = graph.get_operation_by_name("accuracy/temp_sim").outputs[0]

            batches = inpH.batch_iter(list(zip(x1_test, x2_test)), 2 * hp.batch_size, 1, shuffle=False)
            # Collect the predictions here
            predictions = []
            for db in batches:
                x1_dev_b, x2_dev_b = zip(*db)
                batch_sim = sess.run(sim, {input_x1: x1_dev_b, input_x2: x2_dev_b, dropout_keep_prob: 1.0})
                predictions = np.concatenate([predictions, batch_sim])

        return [int(i) for i in predictions]


if __name__ == "__main__":
    sen_a = ["阿贾克斯3-1胜出特温特",
             "据多家泰国媒体3月1日报道，甲米府一女子近日中了1200万泰铢（约合256万元人民币）彩票，而她本人称这是中国新冠疫苗带来的好运。",
             "中小学生原则上不得带手机进校园，手机管理如何防止一刀切？",
             "NBA最新排名！詹姆斯说出施罗德回归意义，勇士有困难，东部大乱"]
    sen_b = ["刚拿下职业新高44分，他又受伤了！三年倒下了9次啊",
             "【环球网报道】路透社刚刚消息，日本厚生劳动省3月2日表示，一名60多岁的女性在接种美国辉瑞公司的新冠疫苗后死亡，死亡原因尚不明确。",
             "作业手机打卡、线上布置任务，家校良性互动只能靠手机？",
             "去年冠军，今年解散？江苏队现在面临的窘境，该由谁来负责？"]
    print(get_sim_lable_mutil(sen_a, sen_b))