import torch

import data_process
from model import PoetryModel
from data_process import optimize_get_data
from configs import opt_shi, opt_ci
from torch.nn import functional as F

import torch as t



class generatePoetry:
    def __init__(self, model, word_to_ix, ix_to_word, device, max_length=100):
        self.model = model
        self.word_to_ix = word_to_ix
        self.ix_to_word = ix_to_word
        self.max_length = max_length
        self.device = device
        self.model.to(device)

    def gen(self,start_words, temperature=0.7, **kwargs):
        """
        给定几个词，根据这几个词接着生成一首完成的诗
        例如，start_words为'海内存知己'，可以生成
        海内存知己，天涯尚未安。
        故人归旧国，新月到新安。
        海气生边岛，江声入夜滩。
        明朝不可问，应与故人看。
        """
        device = self.device
        word2ix, ix2word = self.word_to_ix,self.ix_to_word

        self.model.eval()

        src = [word2ix[word] for word in start_words]
        res = src = [word2ix['<START>']] + src
        max_len = 100

        for _ in range(max_len):
            src = t.tensor(res).to(device)[:, None]
            src_mask = generate_square_subsequent_mask(src.shape[0])
            src_pad_mask = src == word2ix['<PAD>']
            src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
            memory, logits = self.model(src, src_mask.to(device), src_pad_mask.to(device))

            # 温度调节
            logits = logits[-1, 0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, 1).item()

            if next_word == word2ix['<EOP>']:
                break
            res.append(next_word)

        res = [ix2word[_] for _ in res[1:]]
        sen = ''.join(res)
        return sen

    def gen_acrostic_T(self,start_words, temperature=0.7, **kwargs):
        """
        生成藏头诗
        start_words为'深度学习'
        生成：
    	深山高不极，望望极悠悠。
    	度日登楼望，看云上砌秋。
    	学吟多野寺，吟想到江楼。
    	习静多时选，忘机尽处求。
        """
        device = self.device
        word2ix, ix2word = self.word_to_ix,self.ix_to_word

        self.model.eval()

        start_word_len = len(start_words)
        index = 0  # 用来指示已经生成了多少句藏头诗
        src_base = [word2ix[word] for word in start_words]
        res = [word2ix['<START>']] + [src_base[index]]
        index += 1
        max_len = 100

        for _ in range(max_len):
            src = t.tensor(res).to(device)[:, None]
            src_mask = generate_square_subsequent_mask(src.shape[0])
            src_pad_mask = src == len(word2ix) - 1
            src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
            memory, logits = self.model(src, src_mask.to(device), src_pad_mask.to(device))

            # 温度调节
            logits = logits[-1, 0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, 1).item()

            # 如果遇到句号感叹号等，把藏头的词作为下一个句的输入
            if next_word in {word2ix[u'。'], word2ix[u'！'], word2ix['<START>']}:
                # 如果生成的诗歌已经包含全部藏头的词，则结束
                if index == start_word_len:
                    res.append(next_word)
                    break
                # 把藏头的词作为输入，预测下一个词
                res.append(next_word)
                res.append(src_base[index])
                index += 1
            else:
                res.append(next_word)

        res = [ix2word[_] for _ in res[1:]]
        sen=''.join(res)
        return sen

    def generate(self, start_words, max_len=100, temperature=1.0):
        """
        给定首个字 自由生成一首诗
        """
        self.model.eval()
        device=self.device
        word2ix=self.word_to_ix
        ix2word=self.ix_to_word
        src = [word2ix[word] for word in start_words]
        res = src = [word2ix['<START>']] + src

        for _ in range(max_len):
            src = t.tensor(res).to(device)[:, None]
            src_mask = generate_square_subsequent_mask(src.shape[0])
            src_pad_mask = src == len(word2ix) - 1
            src_pad_mask = src_pad_mask.permute(1, 0).contiguous()
            memory, logits = self.model(src, src_mask.to(device), src_pad_mask.to(device))

            # Adjust logits by temperature
            logits = logits[-1, 0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_word = t.multinomial(probs, 1).item()

            if next_word == word2ix['<EOP>']:
                break
            res.append(next_word)

        res = [ix2word[_] for _ in res[1:]]

        #res.remove('<START>')
        sen=''.join(res)
        # print(sen)
        return sen

    def generate_embed(self, embed_words=None, max_len=125, temperature=1.0):
        """
        用于 词嵌入部分
        """
        self.model.eval()
        embed_words=embed_words or []

        res = [self.word_to_ix['<START>']]
        embed_idx = 0  # 嵌入词的索引

        for i in range(max_len):
            src_tensor = torch.tensor(res).to(self.device).unsqueeze(1)
            src_mask = generate_square_subsequent_mask(src_tensor.size(0)).to(self.device)
            src_pad_mask = (src_tensor == self.word_to_ix['<PAD>']).transpose(0, 1).to(self.device)

            memory, logits = self.model(src_tensor, src_mask, src_pad_mask)
            logits = logits[-1, 0] / temperature
            probs = F.softmax(logits, dim=-1)
            next_word = torch.multinomial(probs, 1).item()

            # 按照嵌入词的位置插入指定词语
            if embed_words and embed_idx < len(embed_words) and i % 5 == 0:
                next_word = self.word_to_ix.get(embed_words[embed_idx], next_word)
                embed_idx += 1

            if next_word == self.word_to_ix['<EOP>']:
                break

            res.append(next_word)

        res = [self.ix_to_word[idx] for idx in res[1:]]  # 去掉 <START>
        return ''.join(res)



def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1) # 生成下三角矩阵（下三角全为True，其余位False）
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



_, word_to_ix, ix_to_word = optimize_get_data(opt_ci)
vocab_size = len(word_to_ix)
model_ci = PoetryModel(vocab_size=vocab_size)
model_ci.load_state_dict(torch.load(f=r'./modelsave/ci/ci_199.pth', map_location=device))
model_ci.to(device)

gen_model_ci = generatePoetry(model=model_ci,
                              word_to_ix=word_to_ix,
                              ix_to_word= ix_to_word,
                              max_length=opt_ci.generate_maxlen,
                              device=device)

_,word_to_ix_shi,ix_to_word_shi=optimize_get_data(opt_shi)
vocab_size_shi=len(word_to_ix_shi)
model_shi=PoetryModel(vocab_size=vocab_size_shi)
model_shi.load_state_dict(torch.load(f=r'./modelsave/shi/shi_198.pth',map_location=device))
model_shi.to(device)

gen_model_shi=generatePoetry(model=model_shi,
                             word_to_ix=word_to_ix_shi,
                             ix_to_word=ix_to_word_shi,
                             max_length=opt_shi.generate_maxlen,
                             device=device)




# print(gen_model_ci.gen_acrostic_T(start_words='白色风车', temperature=0.7))
# print(gen_model_ci.gen_acrostic_T(start_words='神州大地', temperature=0.7))
#
# print(gen_model_ci.generate_embed(embed_words=['爱', '古', '蝶', '梦', '魂'], temperature=0.8))
# print(gen_model_ci.generate_embed(embed_words=['瑶', '桂', '香', '媚', '游'], temperature=0.8))
#
# print(gen_model_shi.gen(start_words='海内存知己',temperature=0.7))
#
# print(gen_model_shi.generate(start_words='爱',temperature=0.7))
#




