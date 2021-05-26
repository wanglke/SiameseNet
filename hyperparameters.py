# -*- coding: utf-8 -*-


import os
pwd = os.path.dirname(os.path.abspath(__file__))
class Hyperparamters:
    # basic parameters
    is_char_based = False  # True：在字符级别上进行相似度计算，False：词嵌入
    pretrain_model = "pretrain_model/sgns.weibo.bigram-char"  # pre-trained embeddings file (default: None)
    pretrain_model_format = 'text'
    embedding_dim = 300  # Dimensionality of character embedding (default: 300)
    dropout_keep_prob = 1.0  # Dropout keep probability (default: 1.0)
    l2_reg_lambda = 0.1  # L2 regularizaion lambda (default: 0.0)

    hidden_units = 50  # Number of hidden units (default:50)

    # Train parameters
    match_type = "normal"
    label_type = 'labelA'  # labelA or labelA
    # training_files = [os.path.join(pwd, 'data/souhu/{}'.format(match_type), 'train.txt'),
    #                   os.path.join(pwd, 'data/souhu/{}'.format(match_type), 'valid.txt'),
    #                   os.path.join(pwd, 'data/souhu/round2', '{}.txt'.format(match_type))]

    training_files = os.path.join(pwd, 'data/{}'.format(match_type), 'train_data.tsv')
    model_saved_dir = os.path.join(pwd, 'model_saved', match_type)
    batch_size = 64
    num_epochs = 300
    evaluate_every = 800  # Evaluate model on dev set after this many steps (default: 100)
    checkpoint_every = 800  # Save model after this many steps (default: 100)
    print_step = 100  # every print_step print
    percent_dev = 1


    # Misc Parameters
    allow_soft_placement = True  # Allow device soft device placement
    log_device_placement = False  # Log placement of ops on devices"


    # eval Parameters
    eval_filepath = os.path.join(pwd, 'data/normal/', 'test_data.tsv')
    checkpoint_file = "model_saved/normal/checkpoints/model-24000"  # Load trained model checkpoint (Default: None)
    vocab_filepath = "model_saved/normal/checkpoints/vocab"
    allow_soft_placement = True  # Allow device soft device placement
    log_device_placement = False  # Log placement of ops on devices

if __name__ == '__main__':
    hp = Hyperparamters()


