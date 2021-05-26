from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import re
import numpy as np
import six
from tensorflow.contrib import learn
from tensorflow.python.platform import gfile
from tensorflow.contrib import learn  # pylint: disable=g-bad-import-order
import os
import jieba
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
TOKENIZER_RE = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)


def cut_list_by_size(data, size):
    """
    data: a list
    size: the chunk_size
    """
    return [data[i * size:min((i + 1) * size, len(data))] for i in range(int(len(data) - 1) // size + 1)]
def tokenizer_char(iterator):
    for value in iterator:
        yield [s for s in jieba.cut(value, cut_all=False)]

def tokenizer_word(iterator):
    for value in iterator:
        # by wlk
        # yield TOKENIZER_RE.findall(value)
        yield [s for s in jieba.cut(value, cut_all=False)]


    
class MyVocabularyProcessor(learn.preprocessing.VocabularyProcessor):
    def __init__(self,
               max_document_length,
               min_frequency=0,
               vocabulary=None,
               is_char_based=True):
        if is_char_based:
            tokenizer_fn=tokenizer_char
        else:
            tokenizer_fn=tokenizer_word
        sup = super(MyVocabularyProcessor, self)
        sup.__init__(max_document_length, min_frequency, vocabulary, tokenizer_fn)


    def transform(self, raw_documents):
        """Transform documents to word-id matrix.
        Convert words to ids with vocabulary fitted with fit or the one
        provided in the constructor.
        Args:
          raw_documents: An iterable which yield either str or unicode.
        Yields:
          x: iterable, [n_samples, max_document_length]. Word-id matrix.
        """
        for tokens in self._tokenizer(raw_documents):
            # print(tokens)
        # by wlk
        # for tokens in jieba.cut(raw_documents, cut_all=True):
            word_ids = np.zeros(self.max_document_length, np.int64)
            for idx, token in enumerate(tokens):
                if idx >= self.max_document_length:
                    break
                word_ids[idx] = self.vocabulary_.get(token)
            yield word_ids
