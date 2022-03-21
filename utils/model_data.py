import os
import torch
from utils.constant import *
from utils import data_reader
from models.train_args import *
from models.cnn_text.cnn import CNN
from models.lstm_text.lstm import LSTM
from models.bi_lstm.bilstm import BiLSTM


def make_model_arch(model_type, dataset):
    data = getattr(data_reader, "read_{}".format(dataset))()
    wv_matrix = getattr(data_reader, "wv_matrix_{}".format(dataset))()
    if model_type == ModelType.TEXT_CNN:
        params = args_cnn()
        set_extra_args(data, params)
        params["WV_MATRIX"] = wv_matrix
        model = CNN(**params)
    elif model_type == ModelType.LSTM1:
        params = args_lstm()
        set_extra_args(data, params)
        params["WV_MATRIX"] = wv_matrix
        model = LSTM(**params)
    elif model_type == ModelType.LSTM2:
        params = args_bilstm()
        set_extra_args(data, params)
        params["WV_MATRIX"] = wv_matrix
        model = LSTM(**params)
    elif model_type == ModelType.BiLSTM:
        params = args_bilstm_mr()
        set_extra_args(data, params)
        params["WV_MATRIX"] = wv_matrix
        model = BiLSTM(**params)
    else:
        raise Exception("unsupported model type:{}" / format(model_type))
    return model, params, data


def load_model_data(model_type, dataset, model_path=None):
    if model_path is None:
        model_path = os.path.join(PROJECT_ROOT, getattr(SavedModelPath, "{}_{}".format(model_type, dataset)))
    model, params, data = make_model_arch(model_type, dataset)
    model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
    model.eval()
    return model, params, data

# def cbna_cnn_lstm(dataset):
#     cnn_model, params_cnn, pararent_data = load_model_data(ModelType.TEXT_CNN, dataset)
#     lstm_model, params_lstm, _ = load_model_data(ModelType.LSTM1, dataset)
#     clf_cnn = ClassifierTextCNN(cnn_model, pararent_data["word_to_idx"], params_cnn)
#     clf_lstm = ClassifierLSTM(lstm_model, pararent_data["word_to_idx"], params_lstm)
#     return [clf_cnn, clf_lstm]
#
#
# def cbna_lstm1_lstm2(dataset):
#     lstm_model, params_lstm, pararent_data = load_model_data(ModelType.LSTM1, dataset)
#     lstm_model = lstm_model.cuda(params_lstm["GPU"])
#     clf_lstm = ClassifierLSTM(lstm_model, pararent_data["word_to_idx"], params_lstm)
#
#     lstm_model2, params_lstm2, pararent_data = load_model_data(ModelType.LSTM2, dataset)
#     lstm_model2 = lstm_model2.cuda(params_lstm2["GPU"])
#     clf_lstm2 = ClassifierLSTM(lstm_model2, pararent_data["word_to_idx"], params_lstm2)
#
#     return clf_lstm, clf_lstm2
#
#
# def package_classifier(model_type, data_type):
#     model, params, data = load_model_data(model_type, data_type)
#     if model_type == ModelType.TEXT_CNN:
#         return ClassifierTextCNN(model, data["word_to_idx"], params)
#     elif model_type in [ModelType.BiLSTM, ModelType.LSTM1]:
#         return ClassifierLSTM(model, data["word_to_idx"], params)
#     else:
#         raise Exception("unsupported model type:{}".format(model_type))
