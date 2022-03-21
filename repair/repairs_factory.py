import sys

sys.path.append("../../")
import spacy
import itertools
import copy
import random
from experiments.repairs_generation.paraphraser import *
from repair.sprt import *
from detect.diff_test import kld_pred
from utils.data_reader import load_word2vec
import logging

def load_repair_worker(repair_type, **kwargs):
    if kwargs["word2vec"] is not None:
        word2vec = kwargs["word2vec"]
    else:
        word2vec = load_word2vec(True)
    if repair_type == REPAIRS_TYPE.TRANSLATOR_BASED:
        generator = BFSParaphraser(kwargs["max_depth"])
    elif repair_type == REPAIRS_TYPE.GUIDED_REPLACE:
        logging.debug("loading word2vec")
        generator = GuidedRepair(kwargs["clfs"], word2vec, kwargs["eps"])
    elif repair_type == REPAIRS_TYPE.RANDOM_REPLCE:
        logging.debug("loading word2vec")
        generator = SubstitutionRepair(word2vec)
    else:
        raise Exception("unsupported repair type: {}".format(repair_type))
    return generator


class GuidedRepair():
    def __init__(self, cnn_lstm, word2vec, eps, max_change=4, min_change=3, num_synonyms=10,
                 max_repaires=100):
        self.nlp = spacy.load("en_core_web_sm")
        self.cnn_lstm = cnn_lstm
        self.sim_epsilon = 0.7
        self.word2vec = word2vec
        self.eps = eps
        self.max_repaires = max_repaires
        # self.max_change_ratio = max_change_ratio
        self.max_change = max_change
        self.min_change = min_change
        self.num_synonyms = num_synonyms

    def get_kld(self, sentence):
        kld, (pred1, prob1), (pred2, prob2) = kld_pred(sentence, self.cnn_lstm)
        return abs(kld)

    def document2sents(self, input_text):
        assert isinstance(input_text, str)
        document = self.nlp(input_text)
        sentence_list = []
        for sentence in document.sents:
            sentence_list.append(sentence.text.split(" "))
        return sentence_list

    def sentsList2wordsList(self, sents_list):
        sents = []
        for item in sents_list:
            sents += item
        return sents

    def sort_sents(self, sents_list):
        sent_scores = []
        for sentence in sents_list:
            kld = self.get_kld(sentence)
            sent_scores.append(kld)
        sorted_list = np.argsort(sent_scores)[::-1]
        return [idx for idx in sorted_list]

    def sort_words(self, sentence):
        if len(sentence) == 1:
            sorted_words_idx = [0]
        else:
            word_scores = []
            sent_kld = self.get_kld(sentence)
            for w_idx in range(len(sentence)):
                new_sent = copy.deepcopy(sentence)
                new_sent.pop(w_idx)
                new_kld = self.get_kld(new_sent)
                word_scores.append(
                    sent_kld - new_kld)  # the smaller the new_kld is, the more import the word is since by deleting it, the kld of the sentence likely drops
            sorted_words_idx = np.argsort(word_scores)[::-1]
        return sorted_words_idx

    def replace_word(self, new_word, sent_idx, w_idx, sent_list):
        '''
        replace the w_idx of sent_idx in sent_list with new_word
        :param word:
        :param sent_idx:
        :param w_idx:
        :param sent_list:
        :return:
        '''
        new_sent_list = copy.deepcopy(sent_list)
        new_sent_list[sent_idx][w_idx] = new_word
        return new_sent_list

    def make_new_doc(self, sent_idx, w_idx, old_doc):
        '''
        Substitute-W
        :return: new_sents_list,change_status. change_statue:0 for no change,1 for changed
        '''
        new_docs = []
        target_word = old_doc[sent_idx][w_idx]
        if target_word not in self.word2vec.vocab:
            return new_docs
        words_list = self.word2vec.most_similar(positive=[target_word], topn=self.num_synonyms)
        bugs = [item[0] for item in words_list]
        for bug in bugs:
            new_doc = self.replace_word(bug, sent_idx, w_idx, old_doc)
            new_docs.append(new_doc)
        return new_docs

    def doc_filter(self, docs, ori_kld):
        reapirs = []
        sub_queue = []
        for new_doc in docs:
            new_doc_words_lst = self.sentsList2wordsList(new_doc)
            new_kld = self.get_kld(new_doc_words_lst)
            if new_kld < self.eps:
                reapirs.append(" ".join(new_doc_words_lst).strip())
            if new_kld < ori_kld:
                sub_queue.append(new_doc)
        return sub_queue, reapirs

    def generate_repairs(self, input_text):
        '''
        Notice that in this fashion, we do not return the texts by the default order (descending order of kld)
        :param input_text:
        :return:
        '''
        candts = []
        ori_kld = self.get_kld(input_text)
        if isinstance(input_text, str):
            input_text = input_text.strip(".").split()
        sents_list = self.document2sents(" ".join(input_text))
        sorted_list = self.sort_sents(sents_list)
        newSentsList = copy.deepcopy(sents_list)
        queue = [newSentsList]
        chang_cnt = 0
        for sent_idx in sorted_list:
            sentence = sents_list[sent_idx]
            sorted_words_idx = self.sort_words(sentence)
            for w_idx in sorted_words_idx:
                new_queue = []
                for old_doc in queue:
                    sub_queue = self.make_new_doc(sent_idx, w_idx,
                                                  old_doc)
                    # in this function we control the quality of repairs
                    sub_queue, reapirs = self.doc_filter(sub_queue, ori_kld)
                    new_queue.extend(sub_queue)
                    candts.extend(reapirs)
                    if len(candts) > self.max_repaires:
                        return candts
                if len(new_queue) != 0:
                    queue = new_queue
                    chang_cnt += 1
                if chang_cnt >= self.max_change:
                    return candts
        return candts


class SubstitutionRepair():

    def __init__(self, word2vec, max_replace=4, max_repaires=650):
        self.max_replace = max_replace
        self.max_repaires = max_repaires
        self.word2vec = word2vec
        # todo: add similarity constrians

    def replace_word(self, idx_subset, ori_text):
        new_text = copy.deepcopy(ori_text)
        for idx in idx_subset:
            target_word = ori_text[idx]
            if target_word in self.word2vec.vocab:
                synonyms = self.word2vec.most_similar(positive=[target_word], topn=1)
                new_word = synonyms[0][0]
                assert len(synonyms) == 1
                new_text[idx] = new_word
        return new_text

    def make_subset(self, words_idx, random_seed):
        combs = []
        for num_repplace in range(1, self.max_replace + 1):
            combinations = itertools.combinations(words_idx, num_repplace)
            for subset in combinations:
                combs.append(subset)
        random.seed(random_seed)
        random.shuffle(combs)
        return combs

    def generate_repairs(self, input_text, random_seed=2019):
        '''
        Parameters.
        ===========
        adv_txt: string.
        random_seed: int. used in shuffling the word combinations.
        Return:
        '''
        if isinstance(input_text, str):
            input_text = input_text.strip(".").split()
        words_idx = [i for i in range(len(input_text))]
        candts = []
        combinations = self.make_subset(words_idx, random_seed)
        for subset in combinations:
            new_text = self.replace_word(subset, input_text)
            candts.append(" ".join(new_text).strip())
            if len(candts) > self.max_repaires:
                break
        return candts
