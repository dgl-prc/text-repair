from torch.autograd import Variable
import numpy as np
from utils.model_data import *
from utils.help_func import save_pickle, load_pickle


def _wl_attack_cnn(data, model, params):
    model.eval()
    X, Y = data["test_x"], data["test_y"]
    wl_data = {}
    wl_X = []
    wl_Y = []
    wl_idx = []
    acc = 0
    model = model.cuda(params["GPU"])
    for i in range(0, len(X), params["BATCH_SIZE"]):
        batch_range = min(params["BATCH_SIZE"], len(X) - i)

        batch_x = [[data["word_to_idx"][w] for w in sent] +
                   [params["VOCAB_SIZE"] + 1] * (params["MAX_SENT_LEN"] - len(sent))
                   for sent in X[i:i + batch_range]]

        batch_y = [data["classes"].index(c) for c in Y[i:i + batch_range]]

        batch_x = Variable(torch.LongTensor(batch_x)).cuda(params["GPU"])
        batch_y = Variable(torch.LongTensor(batch_y))

        batch_pred = np.argmax(model(batch_x).cpu().data.numpy(), axis=1)

        idx = np.where(batch_pred != np.array(batch_y))[0]
        wl_X.extend([X[i:i + batch_range][idx] for idx in idx])
        # todo: complete the following code
        # wl_Y.extend(sfs worng label) #
        wl_idx.extend((idx + i).tolist())
        acc += len(batch_x) - len(idx)

    wl_data["wl_x"] = wl_X
    wl_data["wl_y"] = wl_Y
    wl_data["wl_idx"] = wl_idx
    print("Done! Check the test acc:{0:.4f}".format(acc / len(X)))
    return wl_data


def _wl_attck_lstm(data, model, params):
    model.eval()
    if params["GPU"] > 0:
        model = model.cuda(params["GPU"])
    X, Y = data["test_x"], data["test_y"]
    wl_data = {}
    wl_X = []
    wl_Y = []
    wl_idx = []
    i = 0
    for sent, c in zip(X, Y):
        idx_seq = [data["word_to_idx"][w] for w in sent]
        label = data["classes"].index(c)
        idx_seq = torch.LongTensor(idx_seq)
        if params["GPU"] > 0:
            idx_seq = idx_seq.cuda(params["GPU"])
        pred = np.argmax(model(idx_seq).cpu().data.numpy(), axis=1)
        if pred != label:
            wl_X.append(sent)
            wl_Y.append(pred)
            wl_idx.append(i)
        i += 1

    acc = len(X) - len(wl_X)
    wl_data["wl_x"] = wl_X
    wl_data["wl_y"] = wl_Y
    wl_data["wl_idx"] = wl_idx
    print("Done! Check the test acc:{0:.4f}".format(acc / len(X)))
    return wl_data


def cnn_wl_attack():
    model_type = ModelType.TEXT_CNN
    dataset = DataCorpus.IMDB

    model, params, data = load_model_data(model_type, dataset)
    params["GPU"] = 0
    wl_data = _wl_attack_cnn(data, model, params)
    save_path = os.path.join(PROJECT_ROOT, getattr(SavedWLPath, "{}_{}".format(model_type, dataset)))
    save_pickle(save_path, wl_data)


def lstm_wl_attack(model_type, dataset):
    model, params, data = load_model_data(model_type, dataset)
    params["GPU"] = -1
    wl_data = _wl_attck_lstm(data, model, params)
    save_path = os.path.join(PROJECT_ROOT, getattr(SavedWLPath, "{}_{}".format(model_type, dataset)))
    save_pickle(save_path, wl_data)


def do_check(data, model, params, p_data):
    model.eval()
    model = model.cuda(params["GPU"])
    X, Y = data["wl_x"], data["wl_y"]
    wl_X = []
    wl_Y = []
    for sent, c in zip(X, Y):
        idx_seq = [p_data["word_to_idx"][w] for w in sent]
        label = p_data["classes"].index(c)
        idx_seq = torch.LongTensor(idx_seq).cuda(params["GPU"])
        pred = np.argmax(model(idx_seq).cpu().data.numpy(), axis=1)
        if pred != label:
            wl_X.append(sent)
            wl_Y.append(c)

    print("Validation Done! Success rate of attack:{0:.2f}%".format(100 * len(wl_X) / len(X)))


def _check_wl_data():
    model_type = ModelType.LSTM2
    dataset = DataCorpus.IMDB
    wl_data_path = SavedWLPath.lstm_mr
    model, params, data = load_model_data(model_type, dataset)
    wl_data = load_pickle(os.path.join(PROJECT_ROOT, wl_data_path))
    do_check(wl_data, model, params, data)


if __name__ == '__main__':
    lstm_wl_attack(ModelType.BiLSTM, DataCorpus.IMDB)
