import re
import os
from collections import Counter
import numpy as np

_DEFAULT_SEED = 301
_RNG = np.random.RandomState(_DEFAULT_SEED)

CV_DATASETS = {"mr", "subj", "cr", "mpqa"}
DEV_DATASETS = {"sst1", "sst2"}
TREC_DATASETS = {"trec"}


def set_global_seed(seed):
    global _RNG, _DEFAULT_SEED
    _DEFAULT_SEED = seed
    _RNG = np.random.RandomState(seed)


def resolve_train_paths(dataset):
    base = "data"
    if dataset == 'cr':   return os.path.join(base, 'cr', 'cr.all')
    if dataset == 'mpqa': return os.path.join(base, 'mpqa', 'mpqa.all')
    if dataset == 'mr':   return os.path.join(base, 'mr', 'all')
    if dataset == 'sst1': return os.path.join(base, 'sst1', 'shuf_train')
    if dataset == 'sst2': return os.path.join(base, 'sst2', 'shuf_train')
    if dataset == 'subj': return os.path.join(base, 'subj', 'subj.all')
    if dataset == 'trec': return os.path.join(base, 'trec', 'TREC.train.all')
    raise ValueError(f"Unknown dataset: {dataset}")


def resolve_dev_paths(dataset):
    base = "data"
    if dataset == 'sst1': return os.path.join(base, 'sst1', 'stsa.fine.dev')
    if dataset == 'sst2': return os.path.join(base, 'sst2', 'stsa.binary.dev')
    return None


def resolve_test_paths(dataset):
    base = "data"
    if dataset == 'cr':   pass
    if dataset == 'sst1': return os.path.join(base, 'sst1', 'stsa.fine.test')
    if dataset == 'sst2': return os.path.join(base, 'sst2', 'stsa.binary.test')
    if dataset == 'trec': return os.path.join(base, 'trec', 'TREC.test.all')
    return None


def clean_str(sentence, task):
    if task not in ["sst1", "sst2"]:
        sentence = re.sub(r"[^A-Za-z0-9(),!?\"\'\`]", " ", sentence)     
        sentence = re.sub(r"\'s", " \'s", sentence) 
        sentence = re.sub(r"\'ve", " \'ve", sentence) 
        sentence = re.sub(r"n\'t", " n\'t", sentence) 
        sentence = re.sub(r"\'re", " \'re", sentence) 
        sentence = re.sub(r"\'d", " \'d", sentence) 
        sentence = re.sub(r"\'ll", " \'ll", sentence) 
        sentence = re.sub(r",", " , ", sentence) 
        sentence = re.sub(r"!", " ! ", sentence) 
        sentence = re.sub(r"\(", " ( ", sentence) 
        sentence = re.sub(r"\)", " ) ", sentence) 
        sentence = re.sub(r"\?", " ? ", sentence) 
        sentence = re.sub(r"\s{2,}", " ", sentence)
        if task != 'trec':
            sentence = sentence.lower()
    else:
        sentence = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", sentence)   
        sentence = re.sub(r"\s{2,}", " ", sentence)    
        sentence = sentence.lower()
    return sentence


def preprocess_data(data_file, task):
    labels, sentences = [], []
    vocab = Counter()

    with open(data_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                print("[preprocess_data] incorrect format:", line)
                continue
            label, sentence = parts

            label = int(label)
            clear_sentence = clean_str(sentence, task)
            if not clear_sentence: 
                continue
            labels.append(label)
            sentences.append(clear_sentence)
            vocab.update(clear_sentence.split())

    return sentences, labels, vocab


def build_vocab_mapping(vocab, pad_token="<pad>"):
    words = sorted(list(vocab.keys()))
    idx2word = [pad_token] + words   # 0은 pad
    word2idx = {w: i for i, w in enumerate(idx2word)}
    return word2idx, idx2word


def load_bin_vec(bin_name, vocab, dtype=np.float32):
    word_vecs = {}
    with open(bin_name, "rb") as f:
        header = f.readline().decode("utf-8").strip()
        vocab_size, layer_size = map(int, header.split())
        vector_bytes = layer_size * np.dtype(dtype).itemsize
        buffer = np.empty(layer_size, dtype=dtype)

        for _ in range(vocab_size):
            word_bytes = []

            #구분자(space)를 기준으로 바이너리 스트림에서 문자열 추출
            while True:
                ch = f.read(1)
                if ch == b' ':
                    word = b''.join(word_bytes).decode('utf-8', errors='ignore')
                    break
                if ch != b'\n':
                    word_bytes.append(ch)

            #우리 데이터셋에서 등장한 단어만 불러오기
            if word in vocab:
                f.readinto(buffer)
                word_vecs[word] = buffer.copy()
            else:
                f.seek(vector_bytes, 1)
    return word_vecs


# word2vec 벡터에 없는 OOV 단어 랜덤 벡터 채우기
def add_oov_random(word_vecs, vocab, k=300):
    for w in sorted(vocab):
        if w not in word_vecs:
            word_vecs[w] = _RNG.uniform(-0.25, 0.25, k).astype(np.float32)


def build_cv(sentences, labels, cv=10):
    assert len(sentences) == len(labels)
    n = len(sentences)
    splits = _RNG.randint(0, cv, size=n)
    folds = []
    for k in range(cv):
        train_idx = np.nonzero(splits != k)[0].tolist()
        test_idx  = np.nonzero(splits == k)[0].tolist()
        folds.append((train_idx, test_idx))
    return folds

def random_split(labels, ratio=0.1, rng=_RNG):
    n = len(labels)
    idx = np.arange(n)
    rng.shuffle(idx)
    cut = int(round(n * (1 - ratio)))
    return idx[:cut].tolist(), idx[cut:].tolist()

def protocol(task, sentences, labels, dev_file, test_file, dev_ratio=0.1):
    proto = {"task":task, "type":None}

    if task in CV_DATASETS:
        folds = build_cv(sentences, labels, cv=10)
        proto["type"] = "cv"
        proto["folds"] = []
        for k, (train_idx, test_idx) in enumerate(folds):
            proto["folds"].append({
                "fold": k,
                "train_idx": train_idx,
                "dev_idx": test_idx, #CV에서는 test_idx가 fold의 검증역할
                "test": None
            })
        return proto
    
    if task in DEV_DATASETS:
        dev_sents, dev_labels, _ = preprocess_data(dev_file, task)
        test_sents, test_labels, _ = preprocess_data(test_file, task)
        proto["type"] = "dev"
        proto["dev"]  = (dev_sents, dev_labels)
        proto["test"] = (test_sents, test_labels)
        return proto

    if task in TREC_DATASETS:
        train_idx, dev_idx = random_split(labels, ratio=dev_ratio, rng=_RNG)
        test_sents, test_labels, _ = preprocess_data(test_file, task)
        proto["type"] = "trec"
        proto["train_idx"] = train_idx
        proto["dev_idx"]   = dev_idx
        proto["test"] = (test_sents, test_labels)
        return proto

    raise ValueError(f"[protocol] Unsupported task: {task}")


def build_embeddings(word2idx, pretrained=None, k=300, pad_token="<pad>"):
    V = len(word2idx)
    W = _RNG.uniform(-0.25, 0.25, (V, k)).astype(np.float32)
    pad_idx = word2idx.get(pad_token, 0)
    W[pad_idx] = 0.0

    if pretrained:
        any_vec = next(iter(pretrained.values()))
        if any_vec.shape[0] != k:
            raise ValueError(f"[build_embeddings] dim mismatch: k={k}, pretrained={any_vec.shape[0]}")
        for w, idx in word2idx.items():
            if w in pretrained:
                W[idx] = pretrained[w].astype(np.float32)
    return W