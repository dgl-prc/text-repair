from text_attack.textfooler.attack_classification import attack, USE
from models.adapter import PredictorTextFooler
import text_attack.textfooler.criteria as criteria
import numpy as np
import socket
import os



TEXTFOOLER_DATA_PATH = "/home/dgl/project/text_repair/text_attack/textfooler/"

class TextFooler(object):

    def __init__(self, classifier, sim_score_threshold=0.7, import_score_threshold=-1, sim_score_window=15,
                 synonym_num=50,
                 batch_size=1):
        counter_fitting_embeddings_path = os.path.join(TEXTFOOLER_DATA_PATH, "data/counter-fitted-vectors.txt")
        counter_fitting_cos_sim_path = os.path.join(TEXTFOOLER_DATA_PATH, "data/cos_sim_counter_fitting.npy")
        self.sim_score_threshold = sim_score_threshold
        self.import_score_threshold = import_score_threshold
        self.sim_score_window = sim_score_window
        self.synonym_num = synonym_num
        self.batch_size = batch_size
        self.predictor = PredictorTextFooler(classifier)
        self.stop_words_set = criteria.get_stopwords()
        self.use = USE(os.path.join(TEXTFOOLER_DATA_PATH, "catche_path"))
        self.idx2word, self.word2idx = self.__build_dict(counter_fitting_embeddings_path)
        print('Load pre-computed cosine similarity matrix from {}'.format(counter_fitting_cos_sim_path))
        self.cos_sim = np.load(counter_fitting_cos_sim_path)
        print("Cos sim import finished!")

    def __build_dict(self, counter_fitting_embeddings_path):
        # prepare synonym extractor
        # build dictionary via the embedding file
        idx2word = {}
        word2idx = {}
        print("Building vocab...")
        with open(counter_fitting_embeddings_path, 'r') as ifile:
            for line in ifile:
                word = line.split()[0]
                if word not in idx2word:
                    idx2word[len(idx2word)] = word
                    word2idx[word] = len(idx2word) - 1
        return idx2word, word2idx

    def attack(self, text, true_label):
        new_text, num_changed, orig_label, new_label, num_queries = attack(text, true_label, self.predictor,
                                                                           self.stop_words_set,
                                                                           self.word2idx, self.idx2word, self.cos_sim,
                                                                           sim_predictor=self.use,
                                                                           sim_score_threshold=self.sim_score_threshold,
                                                                           import_score_threshold=self.import_score_threshold,
                                                                           sim_score_window=self.sim_score_window,
                                                                           synonym_num=self.synonym_num,
                                                                           batch_size=self.batch_size)
        if orig_label == new_label:
            return None, -1
        return new_text, new_label
