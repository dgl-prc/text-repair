# """
# Gao et al.used the Python autocorrect 0.3.0 package to detect the inputs.
# "Black-box Generation of Adversarial Text Sequences to Evade Deep Learning Classifiers"
# """
import sys
sys.path.append("../")
from utils.help_func import *
from models.adapter import *
from autocorrect import Speller
from baselines.corrector import ScRNNChecker
from utils.data_reader import load_adv_text



class SpellCorrector(object):
    AUTOCORRECT = "ac"  # autocorrect
    SCRNN = "sc"  # scRNN

    def __init__(self, checker_type):
        if checker_type == SpellCorrector.AUTOCORRECT:
            self.checker = Speller()
        elif checker_type == SpellCorrector.SCRNN:
            self.checker = ScRNNChecker()
        else:
            raise Exception("Unsupported type:{}".format(self.checker_type))
        self.checker_type = checker_type

    def correct_string(self, input_str):
        if self.checker_type == SpellCorrector.AUTOCORRECT:
            new_str = self.checker(input_str)
        elif self.checker_type == SpellCorrector.SCRNN:
            new_str = self.checker.correct_string(input_str)
        else:
            raise Exception("Unsupported type:{}".format(self.checker_type))
        return new_str

    def get_name(self):
        if self.checker_type == SpellCorrector.AUTOCORRECT:
            return "autocorrect"
        elif self.checker_type == SpellCorrector.SCRNN:
            return "scrnn"
        else:
            raise Exception("Unsupported type:{}".format(self.checker_type))


def repair_with_speller(dataset, attack_type, model_type, checker_type, adv_size):
    attack_name = str(ATTACK_TYPE(attack_type))
    y_true_lst, y_adv_lst, adv_texts = load_adv_text(attack_type, model_type, dataset)
    classifier = make_classifier(model_type, dataset)
    spell_checker = SpellCorrector(checker_type)
    success_cnt = 0
    adv_texts, y_adv_lst, y_true_lst = adv_texts[:adv_size], y_adv_lst[:adv_size], y_true_lst[:adv_size]
    for sent, adv_label, y_true in zip(adv_texts, y_adv_lst, y_true_lst):
        if isinstance(sent, str):
            str_sent = sent.split()
        else:
            str_sent = sent
        ##########
        # Repair
        ##########
        new_sent = spell_checker.correct_string(" ".join(str_sent)).split()
        new_label = classifier.get_label(new_sent)
        if new_label == y_true:
            success_cnt += 1
    success_rate = 100 * success_cnt / len(y_adv_lst)
    print("Repair method:{}-->({},{},{}) successfully repaired:{:.2f}%({}/{})".format(spell_checker.get_name(),
                                                                                      attack_name,
                                                                                      dataset, model_type, success_rate,
                                                                                      success_cnt,
                                                                                      len(y_adv_lst)))


if __name__ == '__main__':
    # _dataset = sys.argv[1]
    # _attack_type = int(sys.argv[2])
    # _model_type, _checker_type = sys.argv[3], sys.argv[4]
    # repair_with_speller(_dataset, _attack_type, _model_type, _checker_type)
    # _checker_type = "sc"
    # _checker_type = sys.argv[1]
    # _adv_size = int(sys.argv[2])
    # _dataset = DataCorpus.NA
    # for _attack_type in [ATTACK_TYPE.TEXTBUGGER, ATTACK_TYPE.TRANS]:
    #     # for _dataset in [DataCorpus.NA, DataCorpus.MR, DataCorpus.IMDB]:
    #     for _model_type in [ModelType.TEXT_CNN, ModelType.LSTM1]:
    #         repair_with_speller(_dataset, _attack_type, _model_type, _checker_type,_adv_size)
    _adv_size = 1000
    # _dataset = DataCorpus.NA
    _attack_type = ATTACK_TYPE.TEXTFOOLER
    # for _dataset in [DataCorpus.NA, DataCorpus.MR, DataCorpus.IMDB]:
    for _dataset in [DataCorpus.IMDB]:
        for _model_type in [ModelType.TEXT_CNN, ModelType.LSTM1]:
            for _checker_type in [SpellCorrector.AUTOCORRECT, SpellCorrector.SCRNN]:
                repair_with_speller(_dataset, _attack_type, _model_type, _checker_type, _adv_size)
