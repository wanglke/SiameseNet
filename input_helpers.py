import numpy as np
import re
import itertools
from collections import Counter
import numpy as np
import time
import gc
from tensorflow.contrib import learn
from gensim.models.word2vec import Word2Vec
import gzip
from random import random
from preprocess import MyVocabularyProcessor
import sys
from hyperparameters import Hyperparamters as hp
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
class InputHelper(object):
    pre_emb = dict()
    vocab_processor = None
    def cleanText(self, s):
        s = re.sub(r"[^\x00-\x7F]+"," ", s)
        s = re.sub(r'[\~\!\`\^\*\{\}\[\]\#\<\>\?\+\=\-\_\(\)]+',"",s)
        s = re.sub(r'( [0-9,\.]+)',r"\1 ", s)
        s = re.sub(r'\$'," $ ", s)
        s = re.sub('[ ]+',' ', s)
        return s.lower()

    def getVocab(self, vocab_path, max_document_length,filter_h_pad):
        if self.vocab_processor==None:
            print('locading vocab')
            vocab_processor = MyVocabularyProcessor(max_document_length-filter_h_pad,min_frequency=0)
            self.vocab_processor = vocab_processor.restore(vocab_path)
        return self.vocab_processor

    def loadW2V(self,emb_path, type="bin"):
        print("Loading W2V data...")
        num_keys = 0
        if type == "textgz":
            # this seems faster than gensim non-binary load
            for line in gzip.open(emb_path):
                l = line.strip().split()
                st=l[0].lower()
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        if type == "text":
            for line in open(emb_path, encoding='utf-8'):
                l = line.strip().split()
                st = l[0]
                self.pre_emb[st]=np.asarray(l[1:])
            num_keys=len(self.pre_emb)
        else:
            self.pre_emb = Word2Vec.load_word2vec_format(emb_path,binary=True)
            self.pre_emb.init_sims(replace=True)
            num_keys=len(self.pre_emb.vocab)
        print("loaded word2vec len ", num_keys)
        gc.collect()

    def deletePreEmb(self):
        self.pre_emb = dict()
        gc.collect()

    def getTsvData(self, filepath):
        print("Loading training data from "+filepath)
        x1=[]
        x2=[]
        y=[]
        # positive samples from file
        for line in open(filepath,'r',encoding='utf-8'):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if l[0] == 'sen_a' and l[1] == 'sen_b':
                continue
            if random() > 0.5:
                x1.append(l[0].lower())
                x2.append(l[1].lower())
            else:
                x1.append(l[1].lower())
                x2.append(l[0].lower())
            y.append(int(l[2]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)

    def getTsvDataCharBased(self, filepath):
        print("Loading training data from "+filepath)
        x1 =[]
        x2 = []
        y = []
        # positive samples from file
        for line in open(filepath, encoding='UTF-8'):
            l=line.strip().split("\t")
            if len(l)<2:
                continue
            if random() > 0.5:
               x1.append(l[0].lower())
               x2.append(l[1].lower())
            else:
               x1.append(l[1].lower())
               x2.append(l[0].lower())
            y.append(1) #np.array([0,1]))
        # generate random negative samples
        combined = np.asarray(x1+x2)
        shuffle_indices = np.random.permutation(np.arange(len(combined)))
        combined_shuff = combined[shuffle_indices]
        for i in range(len(combined)):
            x1.append(combined[i])
            x2.append(combined_shuff[i])
            y.append(0) #np.array([1,0]))
        return np.asarray(x1),np.asarray(x2),np.asarray(y)


    # add by wlk
    def getJsonDataCharBased(self, filepath):
        x1,x2,y = [], [], []
        # positive samples from file
        for file in filepath:
            print("Loading training data from " + file)
            for line in open(file, encoding='UTF-8'):
                try:
                    source, target, label = '', '', ''
                    data_dict = eval(line)
                    if random() > 0.5:
                        source, target = data_dict['source'], data_dict['target']
                    else:
                        source, target = data_dict['target'], data_dict['source']
                    label = data_dict[hp.label_type]
                except:
                    print("{0} find a illegal data:{}:".format(file,str(line)))
                if source != '' and target != '' and label != '':
                    x1.append(source), x2.append(target), y.append(label)

        return np.asarray(x1), np.asarray(x2), np.asarray(y)



    def getTsvTestData(self, filepath):
        print("Loading testing/labelled data from "+filepath)
        x1 = []
        x2 = []
        y = []
        # positive samples from file
        for line in open(filepath,'r',encoding='utf-8'):
            l=line.strip().split("\t")
            if len(l) < 3:
                continue
            if l[0] == 'sen_a' and l[1] == 'sen_b':
                continue
            x1.append(l[0].lower())
            x2.append(l[1].lower())
            y.append(int(l[2])) # np.array([0,1]))
        return np.asarray(x1), np.asarray(x2), np.asarray(y)
 
    def batch_iter(self, data, batch_size, num_epochs, shuffle=True):
        """
        Generates a batch iterator for a dataset.
        """
        data = np.asarray(data)
        # print(data)
        # print(data.shape)
        data_size = len(data)
        num_batches_per_epoch = int(len(data)/batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]
                
    def dumpValidation(self,x1_text,x2_text,y,shuffled_index,dev_idx,i):
        print("dumping validation "+str(i))
        x1_shuffled=x1_text[shuffled_index]
        x2_shuffled=x2_text[shuffled_index]
        y_shuffled=y[shuffled_index]
        x1_dev=x1_shuffled[dev_idx:]
        x2_dev=x2_shuffled[dev_idx:]
        y_dev=y_shuffled[dev_idx:]
        del x1_shuffled
        del y_shuffled
        with open('validation.txt'+str(i), 'w', encoding='utf-8') as f:
            for text1,text2,label in zip(x1_dev, x2_dev, y_dev):
                f.write(str(label)+"\t"+text1+"\t"+text2+"\n")
            f.close()
        del x1_dev
        del y_dev
    
    # Data Preparatopn
    # ==================================================
    
    
    def getDataSets(self, training_paths, max_document_length, percent_dev, batch_size, is_char_based):
        if is_char_based:
            x1_text, x2_text, y=self.getTsvDataCharBased(training_paths)
            # by wlk
            # x1_text, x2_text, y = self.getJsonDataCharBased(training_paths)
        else:
            x1_text, x2_text, y=self.getTsvData(training_paths)
            # x1_text, x2_text, y = self.getJsonDataCharBased(training_paths)
        # Build vocabulary
        print("Building vocabulary")
        vocab_processor = MyVocabularyProcessor(max_document_length, min_frequency=0, is_char_based=is_char_based)
        vocab_processor.fit_transform(np.concatenate((x2_text, x1_text), axis=0))
        print("Length of loaded vocabulary ={}".format(len(vocab_processor.vocabulary_)))
        i1=0
        train_set=[]
        dev_set=[]
        sum_no_of_batches = 0
        x1 = np.asarray(list(vocab_processor.transform(x1_text)))
        x2 = np.asarray(list(vocab_processor.transform(x2_text)))
        # Randomly shuffle data
        np.random.seed(131)
        shuffle_indices = np.random.permutation(np.arange(len(y)))
        x1_shuffled = x1[shuffle_indices]
        x2_shuffled = x2[shuffle_indices]
        y_shuffled = y[shuffle_indices]
        dev_idx = -1*len(y_shuffled)*percent_dev//100
        del x1
        del x2
        # Split train/test set
        self.dumpValidation(x1_text,x2_text,y,shuffle_indices,dev_idx,1)
        # TODO: This is very crude, should use cross-validation
        x1_train, x1_dev = x1_shuffled[:dev_idx], x1_shuffled[dev_idx:]
        x2_train, x2_dev = x2_shuffled[:dev_idx], x2_shuffled[dev_idx:]
        y_train, y_dev = y_shuffled[:dev_idx], y_shuffled[dev_idx:]
        print("Train/Dev split for {}: {:d}/{:d}".format(training_paths, len(y_train), len(y_dev)))
        sum_no_of_batches = sum_no_of_batches+(len(y_train)//batch_size)
        train_set=(x1_train,x2_train,y_train)
        dev_set = (x1_dev, x2_dev, y_dev)
        gc.collect()
        return train_set,dev_set,vocab_processor,sum_no_of_batches
    
    def getTestDataSet(self, data_path, vocab_path, max_document_length):
        x1_temp, x2_temp, y = self.getTsvTestData(data_path)

        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(x1_temp)))
        x2 = np.asarray(list(vocab_processor.transform(x2_temp)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2, y

    def getTestData(self, sen_a, sen_b, vocab_path, max_document_length):
        # Build vocabulary
        vocab_processor = MyVocabularyProcessor(max_document_length,min_frequency=0)
        vocab_processor = vocab_processor.restore(vocab_path)
        print(len(vocab_processor.vocabulary_))

        x1 = np.asarray(list(vocab_processor.transform(sen_a)))
        x2 = np.asarray(list(vocab_processor.transform(sen_b)))
        # Randomly shuffle data
        del vocab_processor
        gc.collect()
        return x1, x2



