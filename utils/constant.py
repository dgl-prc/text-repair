import sys
sys.path.append("../")
import socket
hostname = socket.gethostname()
if hostname == "amax-vlis2":
    PROJECT_ROOT = "/home/dongguoliang/project/text_repair"
else:
    PROJECT_ROOT = "/home/dgl/project/text_repair"

from utils.pretrained_models_paths import SavedModelPath
MTYPE_SRNN = 'SRN'
MTYPE_LSTM = 'LSTM'
MTYPE_GRU = 'GRU'
# PROJECT_ROOT = "/home/dgl/project/Adversarial-Sampling-for-Repair"
SENTENCE_ENCODER = "text_attack/textbugger/universal-sentence-encoder/"  # universal-sentence-encoder-large
WORD2VEC = "data/word_vectors/glove_word2vec_f.txt"
CHECK_POINT_STEP = 50


class REPAIRS_TYPE:
    RANDOM_REPLCE = "rp"
    GUIDED_REPLACE = "subw"
    TRANSLATOR_BASED = "trans"


class ATTACK_TYPE:
    TRANS = 0
    TEXTBUGGER = 1
    SEAs = 2
    TEXTFOOLER = 3

    def __init__(self, value):
        self.value = value

    def __eq__(self, other):
        if other == self.value:
            return True
        return False

    def __str__(self):
        if self.value == ATTACK_TYPE.TRANS:
            return "tba"
        elif self.value == ATTACK_TYPE.TEXTBUGGER:
            return "textbugger"
        elif self.value == ATTACK_TYPE.SEAs:
            return "seas"
        elif self.TEXTFOOLER == ATTACK_TYPE.TEXTFOOLER:
            return "textfooler"
        else:
            raise Exception("Unsupported attack type:{}".format(self.value))


class DataCorpus:
    IMDB = "imdb"
    NA = "na"
    MR = "mr"


class ModelType:
    TEXT_CNN = "cnn"
    LSTM1 = "lstm"
    BiLSTM = "bilstm"
    LSTM2 = "lstm2"
    FASTTEXT = "fasttext"


class IdPrefix:
    TRAIN = "train_id_"
    ADV = "attack_id_"
    MAKEDATA = "data_id_"


# modeltype, dataset
SavedWLPath = "data/{}/wl/{}_wl.pkl"

TrainingBenignPath = "data/{}/{}_train_benign.pkl"
# dataset, modeltype, attacktype
TrainingAdvDataPath = "data/{}/adv_text/{}/{}_train.pkl"

AdvTrainingModelPath = "models/saved_models/{}/{}/adv_training/{}_{}"


# class SavedWLPath:
#     cnn_mr = "data/mr/wl_test/cnn/wl_data.pkl"
#     cnn_na = "data/na/wl_test/cnn/wl_data.pkl"
#     cnn_imdb = "data/imdb/wl_test/cnn/wl_data.pkl"
#
#     lstm_mr = "data/mr/wl_test/lstm/wl_data.pkl"
#     lstm_na = "data/na/wl_test/lstm/wl_data.pkl"
#     lstm_imdb = "data/imdb/wl_test/lstm/wl_data.pkl"
#
#     bilstm_mr = "data/mr/wl_test/bi_lstm/wl_data.pkl"
#     bilstm_na = "data/na/wl_test/bi_lstm/wl_data.pkl"
#     bilstm_imdb = "data/imdb/wl_test/bi_lstm/wl_data.pkl"
#
#     fasttext_mr = "data/mr/wl_test/fasttext/wl_data.pkl"
#     fasttext_na = "data/na/wl_test/fasttext/wl_data.pkl"
#     fasttext_imdb = "data/imdb/wl_test/fasttext/wl_data.pkl"


class DTD:
    # DTD: Docile Test Data. which means the target model could give a correct prediction.
    # here, we only save the index of data["test_x"]. and the index list is shuffled with random seed 20190807
    # Notice that since the idx are shuffled, we could directly choose top-k elements as a randomly selected subset.
    MR = "data/mr/dtd.pkl"
    NA = "data/na/dtd.pkl"
    IMDB = "data/imdb/dtd.pkl"


class AdvData:

    @classmethod
    def get_adv_data_path(cls, attack_type, model_type, data_type):
        if attack_type == ATTACK_TYPE.TRANS:
            adv_data_class = cls.TBA
        elif attack_type == ATTACK_TYPE.TEXTBUGGER:
            adv_data_class = cls.TEXTBUGGER
        elif attack_type == ATTACK_TYPE.TEXTFOOLER:
            adv_data_class = cls.TEXTFOOLER
        else:
            raise Exception(f"No such attack_type:{attack_type}")
        import os
        save_path = os.path.join(PROJECT_ROOT, getattr(adv_data_class, "{}_{}".format(model_type, data_type)))
        return save_path

    class TBA:
        cnn_mr = "data/mr/adv_text/cnn/tba.pkl"
        cnn_na = "data/na/adv_text/cnn/tba.pkl"
        cnn_imdb = "data/imdb/adv_text/cnn/tba.pkl"

        lstm_mr = "data/mr/adv_text/lstm/tba.pkl"
        lstm_na = "data/na/adv_text/lstm/tba.pkl"
        lstm_imdb = "data/imdb/adv_text/lstm/tba.pkl"

    class TEXTBUGGER:
        cnn_mr = "data/mr/adv_text/cnn/textbugger.pkl"
        cnn_na = "data/na/adv_text/cnn/textbugger.pkl"
        cnn_imdb = "data/imdb/adv_text/cnn/textbugger.pkl"

        lstm_mr = "data/mr/adv_text/lstm/textbugger.pkl"
        lstm_na = "data/na/adv_text/lstm/textbugger.pkl"
        lstm_imdb = "data/imdb/adv_text/lstm/textbugger.pkl"

        bilstm_mr = "data/mr/adv_text/bilstm/textbugger.pkl"
        bilstm_na = "data/na/adv_text/bilstm/textbugger.pkl"
        bilstm_imdb = "data/imdb/adv_text/bilstm/textbugger.pkl"

        fasttext_mr = "data/mr/adv_text/fasttext/textbugger.pkl"
        fasttext_na = "data/na/adv_text/fasttexttextbugger.pkl"
        fasttext_imdb = "data/imdb/adv_text/fasttext/textbugger.pkl"

    class TEXTFOOLER:
        cnn_mr = "data/mr/adv_text/cnn/textfooler.pkl"
        cnn_na = "data/na/adv_text/cnn/textfooler.pkl"
        cnn_imdb = "data/imdb/adv_text/cnn/textfooler.pkl"

        lstm_mr = "data/mr/adv_text/lstm/textfooler.pkl"
        lstm_na = "data/na/adv_text/lstm/textfooler.pkl"
        lstm_imdb = "data/imdb/adv_text/lstm/textfooler.pkl"

        bilstm_mr = "data/mr/adv_text/bilstm/textfooler.pkl"
        bilstm_na = "data/na/adv_text/bilstm/textfooler.pkl"
        bilstm_imdb = "data/imdb/adv_text/bilstm/textfooler.pkl"

        fasttext_mr = "data/mr/adv_text/fasttext/textfooler.pkl"
        fasttext_na = "data/na/adv_text/fasttexttextfooler.pkl"
        fasttext_imdb = "data/imdb/adv_text/fasttext/textfooler.pkl"

    class SEAs:
        cnn_mr = "data/mr/adv_text/cnn/seas.pkl"
        cnn_na = "data/na/adv_text/cnn/seas.pkl"
        cnn_imdb = "data/imdb/adv_text/cnn/seas.pkl"

        lstm_mr = "data/mr/adv_text/lstm/seas.pkl"
        lstm_na = "data/na/adv_text/lstm/seas.pkl"
        lstm_imdb = "data/imdb/adv_text/lstm/seas.pkl"

        bilstm_mr = "data/mr/adv_text/bilstm/seas.pkl"
        bilstm_na = "data/na/adv_text/bilstm/seas.pkl"
        bilstm_imdb = "data/imdb/adv_text/bilstm/seas.pkl"

        fasttext_mr = "data/mr/adv_text/fasttext/seas.pkl"
        fasttext_na = "data/na/adv_text/fasttext/seas.pkl"
        fastext_imdb = "data/imdb/adv_text/fasttext/seas.pkl"


class MergerdAdv:
    class TBA:
        na = "data/na/merged_adv/tba/merged_adv.pkl"
        mr = "data/mr/merged_adv/tba/merged_adv.pkl"
        imdb = "data/imdb/merged_adv/tba/merged_adv.pkl"

    class TEXTBUGGER:
        na = "data/na/merged_adv/textbugger/merged_adv.pkl"
        mr = "data/mr/merged_adv/textbugger/merged_adv.pkl"
        imdb = "data/imdb/merged_adv/textbugger/merged_adv.pkl"


class MergerdAdv:
    class TBA:
        na = "data/na/merged_adv/tba/merged_adv.pkl"
        mr = "data/mr/merged_adv/tba/merged_adv.pkl"
        imdb = "data/imdb/merged_adv/tba/merged_adv.pkl"

    class TEXTBUGGER:
        na = "data/na/merged_adv/textbugger/merged_adv.pkl"
        mr = "data/mr/merged_adv/textbugger/merged_adv.pkl"
        imdb = "data/imdb/merged_adv/textbugger/merged_adv.pkl"


class RepairedCand:
    class TBA:
        na = "data/na/repaired_cand/tba/repaired.pkl"
        mr = "data/mr/repaired_cand/tba/repaired.pkl"
        imdb = "data/imdb/repaired_cand/tba/repaired.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/repaired.pkl"
        mr = "data/mr/repaired_cand/textbugger/repaired.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/repaired.pkl"


class RepairedCandKeys:
    class TBA:
        na = "data/na/repaired_cand/tba/one_depth_repaired_keys.pkl"
        mr = "data/mr/repaired_cand/tba/one_depth_repaired_keys.pkl"
        imdb = "data/imdb/repaired_cand/tba/one_depth_repaired_keys.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/one_depth_repaired_keys.pkl"
        mr = "data/mr/repaired_cand/textbugger/one_depth_repaired_keys.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/one_depth_repaired_keys.pkl"


class SecondRepairedCand:
    class TBA:
        na = "data/na/repaired_cand/tba/second_repaired.pkl"
        mr = "data/mr/repaired_cand/tba/second_repaired.pkl"
        imdb = "data/imdb/repaired_cand/tba/second_repaired.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/second_repaired.pkl"
        mr = "data/mr/repaired_cand/textbugger/second_repaired.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/second_repaired.pkl"


class OnlineSPRTInputData:
    class TBA:
        na = "data/na/repaired_cand/tba/sprt_input.pkl"
        mr = "data/mr/repaired_cand/tba/sprt_input.pkl"
        imdb = "data/imdb/repaired_cand/tba/sprt_input.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/sprt_input.pkl"
        mr = "data/mr/repaired_cand/textbugger/sprt_input.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/sprt_input.pkl"


class ReplaceRepairedCandRnd:  # random replace words with synonyms
    class TBA:
        na = "data/na/repaired_cand/tba/replace_sprt_repaired_rnd.pkl"
        mr = "data/mr/repaired_cand/tba/replace_sprt_repaired_rnd.pkl"
        imdb = "data/imdb/repaired_cand/tba/replace_sprt_repaired_rnd.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/replace_sprt_repaired_rnd.pkl"
        mr = "data/mr/repaired_cand/textbugger/replace_sprt_repaired_rnd.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/replace_sprt_repaired_rnd.pkl"

    class TEXTFOOLER:
        na = "data/na/repaired_cand/textfooler/replace_sprt_repaired_rnd.pkl"
        mr = "data/mr/repaired_cand/textfooler/replace_sprt_repaired_rnd.pkl"
        imdb = "data/imdb/repaired_cand/textfooler/replace_sprt_repaired_rnd.pkl"


class ReplaceRepairedCandGuided:  # guided replace words with synonyms
    class TBA:
        na = "data/na/repaired_cand/tba/replace_sprt_repaired_guid.pkl"
        mr = "data/mr/repaired_cand/tba/replace_sprt_repaired_guid.pkl"
        imdb = "data/imdb/repaired_cand/tba/replace_sprt_repaired_guid.pkl"

    class TEXTBUGGER:
        na = "data/na/repaired_cand/textbugger/replace_sprt_repaired_guid.pkl"
        mr = "data/mr/repaired_cand/textbugger/replace_sprt_repaired_guid.pkl"
        imdb = "data/imdb/repaired_cand/textbugger/replace_sprt_repaired_guid.pkl"

    class TEXTFOOLER:
        na = "data/na/repaired_cand/textfooler/replace_sprt_repaired_guid.pkl"
        mr = "data/mr/repaired_cand/textfooler/replace_sprt_repaired_guid.pkl"
        imdb = "data/imdb/repaired_cand/textfooler/replace_sprt_repaired_guid.pkl"


class RepairStatus:
    NO_REPAIR = "NO_REPAIR"
    REPAIR_FAILED = "REPAIR_FAILED"
    REPAIR_SUCCESS = "REPAIR_SUCCESS"
    NO_BESTLABEL = "NO_BESTLABEL_SPRT"


class CBNT:
    CNN_LSTM = "cnn_lstm"


class DiffJudgeStuff:
    class JudgeTrainData:
        na_cnn_lstm = "data/na/judge_cores/cnn_lstm_data.pkl"
        mr_cnn_lstm = "data/mr/judge_cores/cnn_lstm_data.pkl"
        imdb_cnn_lstm = "data/imdb/judge_cores/cnn_lstm_data.pkl"

    class JudgeModel:
        na_cnn_lstm = "data/na/judge_cores/cnn_lstm.pkl"
        mr_cnn_lstm = "data/mr/judge_cores/cnn_lstm.pkl"
        imdb_cnn_lstm = "data/imdb/judge_cores/cnn_lstm_data.pkl"


class MetricType:
    AJC = "ajc"
    KLD = "kld"
    RBO = "RBO"
    LD = "ld"

# Please use: experiments.step2_search_threshold.search_metric_threshold.get_threshold
# class Threshold:
#     class CNN_LSTM:
#         na_cnn = 0.0288
#         na_lstm = 0.0266
#
#         mr_cnn = 0.0655
#         mr_lstm = 0.1110
#
#         imdb_cnn = 0.0965
#         imdb_lstm = 0.1241
#
#     class CNN_BILSTM:
#         na_cnn = 0.0052
#         na_bilstm = 0.0487
#
#         mr_cnn = 0.0871
#         mr_bilstm = 0.0871
#
#         imdb_cnn = 0.0965
#         imdb_bilstm = 0.1241
#
#     class LSTM_BILSTM:
#         na_lstm = 0.1399
#         na_bilstm = 0.0667
#
#         mr_lstm = 0.1164
#         mr_bilstm = 0.1
#
#         imdb_lstm = 0.0608
#         imdb_bilstm = 0.0384


class AttackDF:
    class BeningData:
        class ComboTwo:
            MR = "experiments/disscuss_white_attack/data/two/mr_bengin.pkl"
            NA = "experiments/disscuss_white_attack/data/two/na_bengin.pkl"
            IMDB = "experiments/disscuss_white_attack/data/two/imdb_bengin.pkl"

        class ComboThree:
            MR = "experiments/disscuss_white_attack/data/three/mr_bengin.pkl"
            NA = "experiments/disscuss_white_attack/data/three/na_bengin.pkl"
            IMDB = "experiments/disscuss_white_attack/data/three/imdb_bengin.pkl"

    class DFAdvData:
        class ComboTwo:
            MR = "experiments/disscuss_white_attack/data/two/mr_adv_df.pkl"
            NA = "experiments/disscuss_white_attack/data/two/na_adv_df.pkl"
            IMDB = "experiments/disscuss_white_attack/data/two/imdb_adv_df.pkl"

        class ComboThree:
            MR = "experiments/disscuss_white_attack/data/three/mr_adv_df.pkl"
            NA = "experiments/disscuss_white_attack/data/three/na_adv_df.pkl"
            IMDB = "experiments/disscuss_white_attack/data/three/imdb_adv_df.pkl"

    class SgAdvData:
        class ComboTwo:
            MR = "experiments/disscuss_white_attack/data/two/mr_adv_sg.pkl"
            NA = "experiments/disscuss_white_attack/data/two/na_adv_sg.pkl"
            IMDB = "experiments/disscuss_white_attack/data/two/imdb_adv_sg.pkl"

        class ComboThree:
            MR = "experiments/disscuss_white_attack/data/three/mr_adv_sg.pkl"
            NA = "experiments/disscuss_white_attack/data/three/na_adv_sg.pkl"
            IMDB = "experiments/disscuss_white_attack/data/three/imdb_adv_sg.pkl"
