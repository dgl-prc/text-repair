from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
import time
import sys
print("start...")
glove_file = sys.argv[1]
tmp_file = "glove_word2vec_f.txt"
_ = glove2word2vec(glove_file, tmp_file)
print("Down!")
print("loading word2vec model...")
stime=time.time()
model = KeyedVectors.load_word2vec_format(tmp_file)
print("{} sencods spent in loading word2vec model".format(int(time.time()-stime)))
