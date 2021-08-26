"""
nn.Transformer와 torchtext를 이용한 언어 번역
======================================================

- 이 튜토리얼에서는 Transformer를 사용하여 처음부터 번역 모델을 학습시키는 방법을 알아보겠습니다.
- 독일어(German)에서 영어(English)로 번역하는 모델을 학습시키기 위해, `Multi30k 
  <http://www.statmt.org/wmt16/multimodal-task.html#task1>`__\ 데이터셋을 사용하겠습니다.

"""


######################################################################
# 데이터 소싱 및 처리
# --------------------
#
# `torchtext 라이브러리 <https://pytorch.org/text/stable/>`__\ 는 언어 번역 모델을 만들 목적으로
# 쉽게 반복할 수 있는 데이터셋을 만드는 유용한 기능이 있습니다.
# 이 예제에서는, torchtext의 내장된 데이터셋을 사용하고, 원본 문장을 토큰화하고,
# 어휘를 구축하고, 토큰들을 tensor로 수치화하는 방법을 보여줍니다.
# 가공되지 않은 원본(source)-대상(target) 쌍을 생성하기 위해서
# `Multi30k dataset from torchtext library <https://pytorch.org/text/stable/datasets.html#multi30k>`__
# 데이터셋을 사용하겠습니다.
#

from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import Multi30k
from typing import Iterable, List


SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'

# Place-holders
token_transform = {}
vocab_transform = {}


# 원본(source)과 대상(target) 언어의 토크나이저를 만듭니다. 패키지 라이브러리를 설치해야 합니다.
# pip install -U spacy
# python -m spacy download en_core_web_sm
# python -m spacy download de_core_news_sm
token_transform[SRC_LANGUAGE] = get_tokenizer('spacy', language='de_core_news_sm')
token_transform[TGT_LANGUAGE] = get_tokenizer('spacy', language='en_core_web_sm')


# 토큰 목록을 생성하는 도움 함수
def yield_tokens(data_iter: Iterable, language: str) -> List[str]:
    language_index = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}

    for data_sample in data_iter:
        yield token_transform[language](data_sample[language_index[language]])

# 특수 기호와 인덱스를 정의합니다.
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
# 토큰들이 어휘에 올바르게 삽입될 수 있도록 해당 인덱스가 순서대로 정렬되었는지 확인합니다.
special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
 
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    # 학습 데이터 반복자
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    # torchtext의 어휘 객체를 생성합니다.
    vocab_transform[ln] = build_vocab_from_iterator(yield_tokens(train_iter, ln),
                                                    min_freq=1,
                                                    specials=special_symbols,
                                                    special_first=True)

# UNK_IDX를 기본 인덱스로 설정하여, 토큰을 찾을 수 없을 때 이 인덱스를 반환합니다.
# 만약 설정하지 않으면, 요청된 토큰을 찾을 수 없을 때 런타임 에러가 발생합니다.
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
  vocab_transform[ln].set_default_index(UNK_IDX)

######################################################################
# Transformer를 이용한 Seq2Seq 네트워크
# -------------------------------------
#
# Transformer는 기계 번역 과제를 해결하기 위한 `“Attention is all you need” 
# <https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf>`__
# 논문에서 소개된 Seq2Seq 모델입니다.
# 아래에서, Transformer를 사용한 Seq2Seq 네트워크를 만들어 볼 것입니다.
# 네트워크는 총 세 부분으로 구성됩니다. 첫번째 파트는 임베딩 레이어입니다.
# 이 레이어는 입력 인덱스 tensor를 이에 대응하는 입력 임베딩 tensor로 변환합니다. 
# 이러한 임베딩은 모델에 입력 토큰의 위치 정보를 제공하기 위한 위치 인코딩과 더해집니다. 
# 두번째 파트는 실제 `Transformer <https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html>`__ 모델입니다.
# 마지막으로, Transformer 모델의 출력은 대상(target) 언어의 각 토큰에 대한 정규화되지 않은 확률값을 제공하는 선형 레이어를 통과합니다.
#


from torch import Tensor
import torch
import torch.nn as nn
from torch.nn import Transformer
import math
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 토큰 임베딩에 위치 인코딩을 추가하여 단어 순서 개념을 도입하는 도움 모듈
class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 5000):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2)* math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding = pos_embedding.unsqueeze(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])

# 입력 인덱스 tensor를 이에 대응하는 토큰 임베딩 tensor로 변환하는 도움 모듈
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, emb_size):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocab_size, emb_size)
        self.emb_size = emb_size

    def forward(self, tokens: Tensor):
        return self.embedding(tokens.long()) * math.sqrt(self.emb_size)

# Seq2Seq 네트워크
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 src_vocab_size: int,
                 tgt_vocab_size: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(emb_size, tgt_vocab_size)
        self.src_tok_emb = TokenEmbedding(src_vocab_size, emb_size)
        self.tgt_tok_emb = TokenEmbedding(tgt_vocab_size, emb_size)
        self.positional_encoding = PositionalEncoding(
            emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                tgt_mask: Tensor,
                src_padding_mask: Tensor,
                tgt_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(self.src_tok_emb(src))
        tgt_emb = self.positional_encoding(self.tgt_tok_emb(trg))
        outs = self.transformer(src_emb, tgt_emb, src_mask, tgt_mask, None, 
                                src_padding_mask, tgt_padding_mask, memory_key_padding_mask)
        return self.generator(outs)

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(
                            self.src_tok_emb(src)), src_mask)

    def decode(self, tgt: Tensor, memory: Tensor, tgt_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(
                          self.tgt_tok_emb(tgt)), memory,
                          tgt_mask)


######################################################################
# 학습 중에는, 예측할 때 모델이 미래의 단어들을 미리 보지 못하게 차단하는 
# 후속 단어 마스크가 필요합니다. 또한 원본(source) 및 대상(target) 패딩 토큰들을 숨기기 위한
# 마스크도 필요합니다. 아래에서, 둘 다 처리할 함수를 정의하겠습니다.
#


def generate_square_subsequent_mask(sz):
    mask = (torch.triu(torch.ones((sz, sz), device=DEVICE)) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask


def create_mask(src, tgt):
    src_seq_len = src.shape[0]
    tgt_seq_len = tgt.shape[0]

    tgt_mask = generate_square_subsequent_mask(tgt_seq_len)
    src_mask = torch.zeros((src_seq_len, src_seq_len),device=DEVICE).type(torch.bool)

    src_padding_mask = (src == PAD_IDX).transpose(0, 1)
    tgt_padding_mask = (tgt == PAD_IDX).transpose(0, 1)
    return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask


######################################################################
# 이제 모델의 매개변수를 정의하고 이를 이용하여 인스턴스를 만듭니다. 아래에서 
# 학습에 사용되는 교차 엔트로피 손실 함수와 옵티마이저를 또한 정의합니다.
#
torch.manual_seed(0)

SRC_VOCAB_SIZE = len(vocab_transform[SRC_LANGUAGE])
TGT_VOCAB_SIZE = len(vocab_transform[TGT_LANGUAGE])
EMB_SIZE = 512
NHEAD = 8
FFN_HID_DIM = 512
BATCH_SIZE = 128
NUM_ENCODER_LAYERS = 3
NUM_DECODER_LAYERS = 3

transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, 
                                 NHEAD, SRC_VOCAB_SIZE, TGT_VOCAB_SIZE, FFN_HID_DIM)

for p in transformer.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

transformer = transformer.to(DEVICE)

loss_fn = torch.nn.CrossEntropyLoss(ignore_index=PAD_IDX)

optimizer = torch.optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)

######################################################################
# 조합(Collation)
# ---------------
#   
# ``데이터 소싱 및 처리`` 섹션에서 볼 수 있듯이, 데이터 반복자는 원본 문자열들의 쌍을 생성합니다.
# 이러한 문자열 쌍들이 이전에 정의된 ``Seq2Seq`` 네트워크에 의해 처리될 수 있도록 배치화된 tensor로 
# 변환해야 합니다. 아래에서는 원본 문자열들의 배치를 모델에 바로 넣을 수 있는 배치 tensor로 변환하는
# collate 함수를 정의합니다.  
#


from torch.nn.utils.rnn import pad_sequence

# 순차적인 연산들을 결합하는 도움 함수
def sequential_transforms(*transforms):
    def func(txt_input):
        for transform in transforms:
            txt_input = transform(txt_input)
        return txt_input
    return func

# 입력 시퀀스 인덱스에 대해 BOS(Beginning Of Sentence)와 EOS(End Of Sentence)를 추가하고 tensor를 만드는 함수
def tensor_transform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]), 
                      torch.tensor(token_ids), 
                      torch.tensor([EOS_IDX])))

# 원본(source) 및 대상(target) 언어 텍스트 변환을 통해 원본 문자열을 tensor 인덱스로 변환
text_transform = {}
for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
    text_transform[ln] = sequential_transforms(token_transform[ln], # 토큰화
                                               vocab_transform[ln], # 수치화
                                               tensor_transform) # BOS/EOS 추가 및 tensor 생성


# 데이터 샘플들을 배치 tensor로 모으는 함수
def collate_fn(batch):
    src_batch, tgt_batch = [], []
    for src_sample, tgt_sample in batch:
        src_batch.append(text_transform[SRC_LANGUAGE](src_sample.rstrip("\n")))
        tgt_batch.append(text_transform[TGT_LANGUAGE](tgt_sample.rstrip("\n")))

    src_batch = pad_sequence(src_batch, padding_value=PAD_IDX)
    tgt_batch = pad_sequence(tgt_batch, padding_value=PAD_IDX)
    return src_batch, tgt_batch
    
######################################################################
# 각 에폭에 대해 호출될 학습 및 평가 루프를 정의합니다.
#

from torch.utils.data import DataLoader

def train_epoch(model, optimizer):
    model.train()
    losses = 0
    train_iter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    train_dataloader = DataLoader(train_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)
    
    for src, tgt in train_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)

        optimizer.zero_grad()

        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        loss.backward()

        optimizer.step()
        losses += loss.item()

    return losses / len(train_dataloader)


def evaluate(model):
    model.eval()
    losses = 0

    val_iter = Multi30k(split='valid', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    val_dataloader = DataLoader(val_iter, batch_size=BATCH_SIZE, collate_fn=collate_fn)

    for src, tgt in val_dataloader:
        src = src.to(DEVICE)
        tgt = tgt.to(DEVICE)

        tgt_input = tgt[:-1, :]

        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = create_mask(src, tgt_input)

        logits = model(src, tgt_input, src_mask, tgt_mask,src_padding_mask, tgt_padding_mask, src_padding_mask)
        
        tgt_out = tgt[1:, :]
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), tgt_out.reshape(-1))
        losses += loss.item()

    return losses / len(val_dataloader)

######################################################################
# 이제 모델을 학습할 때 필요한 모든 요소가 마련되었습니다. 학습을 진행해봅시다!
#

from timeit import default_timer as timer
NUM_EPOCHS = 18

for epoch in range(1, NUM_EPOCHS+1):
    start_time = timer()
    train_loss = train_epoch(transformer, optimizer)
    end_time = timer()
    val_loss = evaluate(transformer)
    print((f"Epoch: {epoch}, Train loss: {train_loss:.3f}, Val loss: {val_loss:.3f}, "f"Epoch time = {(end_time - start_time):.3f}s"))


# 탐욕 알고리즘을 사용하여 출력 시퀀스를 생성하는 함수
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generate_square_subsequent_mask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys


# 실제로 입력 문장을 대상(target) 언어로 번역하는 함수
def translate(model: torch.nn.Module, src_sentence: str):
    model.eval()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    return " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")


######################################################################
#

print(translate(transformer, "Eine Gruppe von Menschen steht vor einem Iglu ."))


######################################################################
# 참고문헌
# ----------
#
# 1. Attention is all you need 논문.
#    https://papers.nips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf
# 2. 주석이 달린 transformer. https://nlp.seas.harvard.edu/2018/04/03/attention.html#positional-encoding
