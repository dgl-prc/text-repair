# coding:utf8
import sys

sys.path.append("../../")
import torch
import requests
import uuid
import pickle
from utils.constant import *
from text_attack.abs_attacker import TextAttacker
import time
import http.client
import hashlib
import json
import urllib
import random
from collections import defaultdict
import traceback
from utils.tranlators_api_keys import TranslatorAPIKeys


class MSTranslatorAttacker(TextAttacker):
    def __init__(self, target_model):
        self.subscriptionKey = TranslatorAPIKeys.ms_key2
        self.base_url = 'https://api.cognitive.microsofttranslator.com'
        self.path = '/translate?api-version=3.0'
        self.headers = {
            'Ocp-Apim-Subscription-Key': self.subscriptionKey,
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        self.all_langs = self.get_all_support_lang()
        self.model = target_model
        self.time_sleep = 10  # seconds
        self.sleep_cnt = 0

    def get_all_support_lang(self):
        base_url = 'https://api.cognitive.microsofttranslator.com'
        path = '/languages?api-version=3.0'
        constructed_url = base_url + path
        headers = {
            'Content-type': 'application/json',
            'X-ClientTraceId': str(uuid.uuid4())
        }
        request = requests.get(constructed_url, headers=headers)
        response = request.json()
        support_code = []
        support_name = []
        for i, key in enumerate(response["dictionary"].keys()):
            if key != "en":
                support_code.append(key)
                support_name.append(response["dictionary"][key]["name"])
        rst = {"code": support_code, "name": support_name}
        return rst

    def make_targets_langs(self, target_langs):
        if not isinstance(target_langs, list):
            target_langs = [target_langs]
        params = '&to='.join(target_langs)
        params = '&to=' + params
        return params

    def make_body(self, input_texts):
        assert isinstance(input_texts, list)
        body = [{'text': input_text} for input_text in input_texts]
        return body

    def wait(self):
        print("Too many requests sent. Waiting for a moment......")
        if self.sleep_cnt > 0:
            self.sleep_cnt += 1
            self.time_sleep += 5
        time.sleep(self.time_sleep)

    def translate2pivots(self, pivot_langs, input_text):
        params = self.make_targets_langs(pivot_langs)
        body = self.make_body(input_text)
        constructed_url = self.base_url + self.path + params
        request = requests.post(constructed_url, headers=self.headers, json=body)
        response = request.json()
        pivot_results = {}
        if isinstance(response, dict):
            if str(response["error"]["code"]).startswith("429"):
                self.wait()
                return self.translate2pivots(pivot_langs, input_text)
            else:
                raise Exception(response)
        else:
            for result in response[0]["translations"]:
                language = result["to"]
                text = result["text"]
                pivot_results[language] = text
        return pivot_results

    def pivots2en(self, pivot_results):
        num_text = len(pivot_results)
        if num_text > 25:
            raise Exception("Too many text submitted!")
        params = self.make_targets_langs("en")
        body = self.make_body(pivot_results)
        constructed_url = self.base_url + self.path + params
        request = requests.post(constructed_url, headers=self.headers, json=body)
        response = request.json()
        paraphrases = []
        if isinstance(response, dict):  # only the error message is dict type
            if str(response["error"]["code"]).startswith("429"):
                self.wait()
                return self.pivots2en(pivot_results)
            else:
                raise Exception(response)
        else:
            for item in response:
                text = item["translations"][0]["text"]
                paraphrases.append(text)
            return paraphrases

    def paraphrase_text(self, input_text, *args):
        pivot_langs = args[0]
        if not isinstance(input_text, list):
            input_text = [input_text]
        pivot_results = self.translate2pivots(pivot_langs, input_text)
        pivot_text = [pivot_results[lang] for lang in pivot_results.keys()]
        start = 0
        paraphrases = []
        for i in range(10):
            small_paraphrases = self.pivots2en(pivot_text[start:start + 5])
            start += 5
            paraphrases += small_paraphrases
        return paraphrases

    def attack(self, input_text, *args):
        self.sleep_cnt = 0
        self.time_sleep = 10
        orig_pred = self.model.get_label(input_text)[0]
        texts = self.paraphrase_text(input_text, self.all_langs["code"])
        preds = self.model.get_label(texts)
        adv_texts = []
        for i, adv_pred in enumerate(preds):
            if adv_pred != orig_pred:
                adv_texts.append((texts[i], adv_pred))
        return adv_texts, orig_pred


class BaiDuTranslatorAttacker(TextAttacker):
    def __init__(self, target_model=None, appid_num=0):
        self.id_key_pairs = TranslatorAPIKeys.baidu_ID_key_pairs
        self.homeUrl = 'api.fanyi.baidu.com'
        self.requestUrl = '/api/trans/vip/translate'
        self.models = target_model
        self.all_langs = self.get_all_support_lang()
        self.appid = self.id_key_pairs[appid_num]["id"]
        self.secretkey = self.id_key_pairs[appid_num]["key"]
        self.time_sleep = 5  # seconds
        self.sleep_cnt = 0
        self.max_tries = 10

    def get_all_support_lang(self):
        support_langs = {'zh': "中文",
                         'yue': "粤语",
                         'jp': "日语",
                         'kor': "韩语",
                         'fra': "法语",
                         'spa': "西班牙语",
                         'th': "泰语",
                         'ara': "阿拉伯语",
                         'ru': "俄语",
                         'pt': "葡萄牙语",
                         'de': "德语",
                         'it': "意大利语",
                         'el': "希腊语",
                         'nl': "荷兰语",
                         'pl': "波兰语",
                         'bul': "保加利亚语",
                         'est': "爱沙尼亚语",
                         'dan': "丹麦语",
                         'fin': "芬兰语",
                         'cs': "捷克语",
                         'rom': "罗马尼亚语",
                         'slo': "斯洛文尼亚语",
                         'swe': "瑞典语",
                         'hu': "匈牙利语",
                         'vie': "越南语"}
        langs = [key for key in support_langs]
        return sorted(langs)  # ordered by alphabet

    def wait(self, error_message, error_code):
        if error_code == "54004":
            print("{}.".format(error_message))
            self.reset_id_key()
        else:
            print("{}. Waiting for about {} seconds......".format(error_message, self.time_sleep))
            time.sleep(self.time_sleep)
            self.sleep_cnt += 1
            self.time_sleep += 10
            if self.sleep_cnt > 2:
                self.reset_id_key()
                self.reset_wait_time()

    def reset_id_key(self):
        idx = random.randint(0, len(self.id_key_pairs) - 1)  # note a<=n<=b
        self.appid = self.id_key_pairs[idx]["id"]
        self.secretkey = self.id_key_pairs[idx]["key"]
        print("Change to account: {}".format(self.appid))

    def reset_wait_time(self):
        self.time_sleep = 5  # seconds
        self.sleep_cnt = 0

    def make_requesturl(self, q, fromLang, toLang):
        salt = random.randint(32768, 65536)
        sign = self.appid + q + str(salt) + self.secretkey
        sign = hashlib.md5(sign.encode()).hexdigest()
        myurl = self.requestUrl + '?appid=' + self.appid + '&q=' + urllib.parse.quote(
            q) + '&from=' + fromLang + '&to=' + toLang + '&salt=' + str(
            salt) + '&sign=' + sign
        return myurl

    def translate2pivots(self, pivot_langs, input_text):
        fromLang = "en"
        pivot_results = {}
        for toLang in pivot_langs:
            cnt = 0
            while True:
                cnt += 1
                if cnt > self.max_tries:
                    print("jump this pivot:{}  Input text:{}".format(toLang, input_text))
                    break
                requestURL = self.make_requesturl(input_text, fromLang, toLang)
                httpClient = http.client.HTTPConnection(self.homeUrl)
                try:
                    httpClient.request('GET', requestURL)
                    response = httpClient.getresponse()
                    jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
                    js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
                except json.decoder.JSONDecodeError as e:
                    httpClient.close()
                    print(e)
                    print(jsonResponse)
                    break
                except Exception as e:
                    httpClient.close()
                    error_msg = "{}:translate2pivots --> from 'en' ---> '{}' ".format(e, toLang)
                    self.wait(error_msg, "-1111")
                    # return self.translate2pivots(pivot_langs, input_text)
                    continue
                httpClient.close()
                if 'error_code' in js.keys():
                    self.wait(js['error_msg'], js['error_code'])
                    continue
                    # return self.translate2pivots(pivot_langs, input_text)
                else:
                    dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
                    pivot_results[toLang] = dst
                    break
            time.sleep(1)
        return pivot_results

    def pivots2en(self, pivot_results):
        toLang = "en"
        paraphrases = {}
        for fromLang in pivot_results.keys():
            cnt = 0
            while True:
                cnt += 1
                input_text = pivot_results[fromLang]
                if cnt > self.max_tries:
                    print("jump this pivot:{}. Input text:{}".format(fromLang, input_text))
                    break
                requestURL = self.make_requesturl(input_text, fromLang, toLang)
                httpClient = http.client.HTTPConnection(self.homeUrl)
                try:
                    httpClient.request('GET', requestURL)
                    response = httpClient.getresponse()
                    jsonResponse = response.read().decode("utf-8")  # 获得返回的结果，结果为json格式
                    js = json.loads(jsonResponse)  # 将json格式的结果转换字典结构
                except json.decoder.JSONDecodeError as e:
                    httpClient.close()
                    print(e)
                    print(jsonResponse)
                    break
                except Exception as e:
                    httpClient.close()
                    error_msg = "{}:pivots2en --> from '{}' ---> 'en' ".format(e, fromLang)
                    self.wait(error_msg, "-11111")
                    continue
                    # return self.pivots2en(pivot_results)
                httpClient.close()
                if 'error_code' in js.keys():
                    self.wait(js['error_msg'], js['error_code'])
                    continue
                    # return self.pivots2en(pivot_results)
                else:
                    dst = str(js["trans_result"][0]["dst"])  # 取得翻译后的文本结果
                    paraphrases[fromLang] = dst
                    break
            time.sleep(1)
        return paraphrases

    def filter_invalid_paraphrase(self, input_text, paraphrases):
        new_input_text = input_text.strip().lower()
        pivots = [pivot for pivot in paraphrases]
        for pivot in pivots:
            para = paraphrases[pivot]
            if new_input_text == para.strip().lower():
                del paraphrases[pivot]

    def paraphrase_text(self, input_text, *args):
        self.reset_wait_time()
        pivot_langs = args[0]
        pivot_results = self.translate2pivots(pivot_langs, input_text)
        paraphrases = self.pivots2en(pivot_results)
        self.filter_invalid_paraphrase(input_text, paraphrases)
        remove_redundant = defaultdict(list)
        for lang in paraphrases.keys():
            assert paraphrases[lang] != input_text  # ensure valid parapharase
            remove_redundant[paraphrases[lang]].append(lang)
        return remove_redundant

    def filter_bad_adv(self, texts_dict, orig_pred):
        texts = []
        langs = []
        for item in texts_dict.items():
            texts.append(item[0])
            langs.append(item[1])
        adv_texts = []
        for i, text in enumerate(texts):
            lang = langs[i]
            pred = self.model.get_label(text)[0]
            if pred != orig_pred:
                adv_texts.append((text, pred, orig_pred, lang))
        return adv_texts

    def attack(self, input_text):
        orig_pred = self.model.get_label(input_text)[0]
        sentence = " ".join(input_text)
        for pivot in self.all_langs.keys():
            texts_dict = self.paraphrase_text(sentence, [pivot])  # dict:{sent:[lang1,lang2...]}
            adv_texts = self.filter_bad_adv(texts_dict, orig_pred)
            if len(adv_texts) > 0:
                assert len(adv_texts) == 0
                return adv_texts[0]
            return []


class SaveMoneyTBA(BaiDuTranslatorAttacker):
    def __init__(self, target_model):
        super(SaveMoneyTBA, self).__init__(target_model)

    def set_target_models(self, target_cnn, target_lstm):
        self.target_cnn = target_cnn
        self.target_lstm = target_lstm

    def paraphrase_text(self, input_text, *args):
        self.reset_wait_time()
        pivot_langs = args[0]
        pivot_results = self.translate2pivots(pivot_langs, input_text)
        paraphrases = self.pivots2en(pivot_results)
        return list(paraphrases.items())[0][1]

    def attack(self, input_text):
        orig_p_cnn = self.target_cnn.get_label(input_text)
        orig_p_lstm = self.target_lstm.get_label(input_text)
        assert orig_p_cnn == orig_p_lstm
        sentence = " ".join(input_text)
        cnn_adv = ()
        lstm_adv = ()
        cnn_cnt = 1  # the number of pivot languages used
        lstm_cnt = 1  # the number of pivot languages used

        cnn_stop = False
        lstm_stop = False
        for pivot in self.all_langs.keys():
            try:
                text = self.paraphrase_text(sentence, [pivot])
                p_cnn = self.target_cnn.get_label(text.split())
                p_lstm = self.target_lstm.get_label(text.split())
            except Exception as e:
                raise Exception(
                    "Attack Error! pivot:{},\nErrorMsg:{}\nTrace:{}".format(pivot, str(e), traceback.format_exc()))

            if orig_p_cnn != p_cnn and not cnn_stop:
                cnn_adv = (text.split(), p_cnn, orig_p_cnn)
                cnn_stop = True

            if orig_p_lstm != p_lstm and not lstm_stop:
                lstm_adv = (text.split(), p_lstm, orig_p_lstm)
                lstm_stop = True

            if cnn_stop and lstm_stop:
                break

            if not cnn_stop:
                cnn_cnt += 1
            if not lstm_stop:
                lstm_cnt += 1
        if not cnn_stop:  # if eventually not success
            cnn_cnt -= 1
        if not lstm_stop:
            lstm_cnt -= 1
        return (cnn_adv, cnn_cnt), (lstm_adv, lstm_cnt)
