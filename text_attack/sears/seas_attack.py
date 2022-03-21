from text_attack.sears import onmt_model
import spacy
from text_attack.sears import paraphrase_scorer
from text_attack.sears import replace_rules
from text_attack.abs_attacker import TextAttacker
from utils.help_func import clean_data_for_look
import editdistance
from utils.time_util import current_timestamp


class SEAsAttack(TextAttacker):
    def __init__(self, model, topk=100, edit_distance_cutoff=5, threshold=-10, max_iter=1000, gpu=0):
        self.ps = paraphrase_scorer.ParaphraseScorer(gpu_id=gpu)
        self.nlp = spacy.load('en')
        self.tokenizer = replace_rules.Tokenizer(self.nlp)
        self.classifier = model
        self.topk = topk
        self.edit_distance_cutoff = edit_distance_cutoff
        self.threshold = threshold
        self.max_iter = max_iter

    def paraphrase_text(self, input_text):
        instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in self.nlp.tokenizer(input_text)]),
                                                  only_upper=False)

        paraphrases = self.ps.generate_paraphrases(instance_for_onmt, topk=self.topk,
                                                   edit_distance_cutoff=self.edit_distance_cutoff,
                                                   threshold=self.threshold)

        texts = self.tokenizer.clean_for_model(self.tokenizer.clean_for_humans([x[0] for x in paraphrases]))
        return texts
        # better_texts = [text for text in texts if
        #                 editdistance.eval(text.split(), instance_for_onmt.split()) < self.edit_distance_cutoff]
        # return better_texts

    def attack(self, input_text):
        print(current_timestamp(), "ORI TEXT >>> ", " ".join(input_text))
        assert type(input_text) == list
        orig_pred = self.classifier.get_label(input_text)
        queue = [" ".join(input_text)]
        iter_cnt = 0
        while len(queue) > 0:
            ori_text = queue.pop(0)
            sim_texts = self.paraphrase_text(ori_text)
            for candidate in sim_texts:
                if len(candidate.split()) == 0: continue
                pred = self.classifier.get_label(candidate.split())
                if pred != orig_pred:
                    print(current_timestamp(), "ADV TEXT >>> ", candidate)
                    return candidate.split(), orig_pred
                else:
                    queue.append(candidate)
                iter_cnt += 1
                if iter_cnt > self.max_iter:
                    return None, -1
