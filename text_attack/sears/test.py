import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
import sys
sys.path.append('../../') # noqa
from text_attack.sears import paraphrase_scorer
from text_attack.sears  import onmt_model
import numpy as np
import spacy
nlp = spacy.load('en')
from text_attack.sears  import replace_rules
tokenizer = replace_rules.Tokenizer(nlp)
ps = paraphrase_scorer.ParaphraseScorer(gpu_id=-1)
topk=100
threshold=-10
instance="Despite Russia's boasts, deal with China leaves it playing second fiddle"
instance_for_onmt = onmt_model.clean_text(' '.join([x.text for x in nlp.tokenizer(instance)]), only_upper=False)
paraphrases = ps.generate_paraphrases(instance_for_onmt, topk=topk, edit_distance_cutoff=4, threshold=threshold)
for p in paraphrases:
    print(p)
