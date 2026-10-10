import logging
import os

import numpy as np
import torch

import config
from beam_decoder import beam_search
from model.tf_model import make_model
from tools.tokenizer_utils import chinese_tokenizer_load, english_tokenizer_load

logging.basicConfig(
    format='%(asctime)s-%(name)s-%(levelname)s-%(message)s-%(funcName)s:%(lineno)d',
    level=logging.INFO
)


def _make_model():
    return make_model(
        config.src_vocab_size,
        config.tgt_vocab_size,
        config.n_layers,
        config.d_model,
        config.d_ff,
        config.n_heads,
        config.dropout
    )


def _decode(src, model, sp_chn):
    with torch.no_grad():
        src_mask = (src != config.padding_idx).unsqueeze(-2)
        decode_result, _ = beam_search(
            model,
            src,
            src_mask,
            config.max_len,
            config.padding_idx,
            config.bos_idx,
            config.eos_idx,
            config.beam_size,
            config.device
        )
        return sp_chn.decode_ids(decode_result[0][0])


def translate(src, model):
    """用训练好的模型进行预测单句。"""
    model.load_state_dict(torch.load(config.test_model_path, map_location=config.device))
    model.eval()
    return _decode(src, model, chinese_tokenizer_load())


class Translator:
    """加载一次翻译模型和分词器，供网页等长时间运行的入口复用。"""

    def __init__(self, model_path=None):
        self.model_path = model_path or config.test_model_path
        if not os.path.isfile(self.model_path):
            raise FileNotFoundError(f"模型权重不存在: {self.model_path}")

        self.model = _make_model()
        self.model.load_state_dict(torch.load(self.model_path, map_location=config.device))
        self.model.eval()
        self.sp_eng = english_tokenizer_load()
        self.sp_chn = chinese_tokenizer_load()

    def translate(self, sent):
        sent = sent.strip()
        if not sent:
            raise ValueError("请输入要翻译的英文句子。")

        src_tokens = [[
            self.sp_eng.bos_id(),
            *self.sp_eng.EncodeAsIds(sent),
            self.sp_eng.eos_id()
        ]]
        batch_input = torch.LongTensor(np.array(src_tokens)).to(config.device)
        return _decode(batch_input, self.model, self.sp_chn)


_translator = None


def one_sentence_translate(sent):
    """翻译单句英文，并复用已加载的模型。"""
    global _translator
    if _translator is None:
        _translator = Translator()
    return _translator.translate(sent)


def translate_example():
    """交互式单句翻译示例。"""
    while True:
        try:
            sent = input("请输入英文句子进行翻译（输入 q 退出）：")
        except EOFError:
            break
        if sent.strip().lower() in {"q", "quit", "exit"}:
            break
        try:
            print("翻译结果：", one_sentence_translate(sent))
        except ValueError as error:
            print(error)


if __name__ == "__main__":
    import warnings

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    warnings.filterwarnings("ignore")
    translate_example()
