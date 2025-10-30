import argparse
import torch
import random
from torch.utils.data import Dataset, DataLoader
from torch.nn import functional as F
import numpy as np
from tqdm import tqdm
from utils import *
from model import SentenceCNN

class CnnDataset(Dataset):
    """
    * 문장 좌/우로 0을 pad_n개 패딩
    * 토큰이 word2idx에 있을 때에만 추가 (OOV 드랍)
    * 본문은 max_l로 자르고 최종 길이는 max_l + 2*pad_n 고정
    """
    def __init__(self, sentences, labels, word2idx, max_l, pad_n=4, pad_id=0):
        self.sentences = sentences
        self.labels = labels
        self.word2idx = word2idx
        self.max_l = max_l
        self.pad_n = pad_n
        self.pad_id = pad_id
        self.total_len = self.max_l + 2 * self.pad_n
        
    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        s = self.sentences[idx]
        y = self.labels[idx]

        raw_toks = s.strip().split()
        kept = []
        for tok in raw_toks:
            if tok in self.word2idx:
                kept.append(self.word2idx[tok])

        ids = [self.pad_id] * self.pad_n + kept
        if len(ids) > self.pad_n + self.max_l:
            ids = ids[: self.pad_n + self.max_l]
        if len(ids) < self.total_len:
            ids += [self.pad_id] * (self.total_len - len(ids))

        return torch.tensor(ids, dtype=torch.long), torch.tensor(y, dtype=torch.long)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    total_correct, total_seen = 0, 0
    for x, y in loader:
        x = x.to(device); y = y.to(device)
        logits = model(x)
        preds = logits.argmax(dim=1)
        total_correct += (preds == y).sum().item()
        total_seen += y.size(0)
    return {"acc": total_correct / max(1, total_seen)}


def build_model(mode, n_labels, word2idx, pretrained, hyperparams):
    return SentenceCNN(
        mode=mode,
        n_labels=n_labels,
        word2idx=word2idx,
        pretrained=pretrained,
        dim=hyperparams['dim'],
        num_feature_maps=hyperparams['num_feature_maps'],
        filter_sizes=hyperparams['filter_sizes'],
        p_dropout=hyperparams['p_dropout'],
        padding_idx=hyperparams['padding_idx']
    )


def train(train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels,
          model, hyperparameters, word2idx):

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("[device]", device)
    model = model.to(device)

    max_l = max(len(s.strip().split()) for s in train_sents)
    pad_n = 4 #[3,4,5] → max=5
    print(f"[max_l={max_l}, pad_n={pad_n}]")

    train_ds = CnnDataset(train_sents, train_labels, word2idx, max_l, pad_n)
    dev_ds   = CnnDataset(dev_sents,   dev_labels,   word2idx, max_l, pad_n) if dev_sents  else None
    test_ds  = CnnDataset(test_sents,  test_labels,  word2idx, max_l, pad_n) if test_sents else None

    train_loader = DataLoader(train_ds, batch_size=hyperparameters['batch_size'], shuffle=True)
    dev_loader   = DataLoader(dev_ds,   batch_size=hyperparameters['batch_size']) if dev_ds else None
    test_loader  = DataLoader(test_ds,  batch_size=hyperparameters['batch_size']) if test_ds else None

    print("[Dataset] total_len", train_ds.total_len, "max_l(train)", max_l, "pad_n", pad_n)
    print("[Loader] train batches:", len(train_loader))
    if dev_loader:  print("[Loader] dev batches:", len(dev_loader))
    if test_loader: print("[Loader] test batches:", len(test_loader))


    optimizer = torch.optim.Adadelta(model.parameters(), rho=0.95)
    criterion = torch.nn.CrossEntropyLoss()

    # 학습 루프
    best_dev = -1.0
    best_state = None
    patience_left = 5

    for ep in range(25):
        model.train()
        correct, seen = 0, 0
        
        for x, y in tqdm(train_loader, desc=f"train epoch {ep:02d}"):
            x = x.to(device); y = y.to(device) 
            
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            if hasattr(model, "fc"):
                with torch.no_grad():
                    norm = model.fc.weight.norm(2)
                    if norm > 3.0:
                        model.fc.weight.mul_(3.0 / norm)
    
            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            seen += y.size(0)

        train_acc = correct / max(1, seen)
        log = f"[epoch {ep:02d}] train_acc={train_acc:.4f}"

        # dev O → Early Stopping
        if dev_loader is not None:
            dev_metric = evaluate(model, dev_loader, device)
            log += f" dev_acc={dev_metric['acc']:.4f}"

            if dev_metric["acc"] > best_dev:
                best_dev = dev_metric["acc"]
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                patience_left = 5
            else:
                patience_left -= 1
            print(log)

            if patience_left <= 0:
                print(f"[early stop] patience exhausted at epoch {ep}.")
                break
        else:
            print(log)

    if best_state is not None:
        model.load_state_dict(best_state)

    final_dev_metric  = evaluate(model, dev_loader,  device) if dev_loader  is not None else {"acc": 0.0}
    final_test_metric = evaluate(model, test_loader, device) if test_loader is not None else None

    return final_dev_metric, final_test_metric

def find_nearest_neighbors_multi(model, word2idx, idx2word, word_list, topk=10, padding_idx=0, print_table=True):
    W_static = model.emb_static.weight.detach().cpu()       # (V, D)
    W_non    = model.emb_nonstatic.weight.detach().cpu()    # (V, D)
    W_static = F.normalize(W_static, dim=1)                 # (V, D)
    W_non    = F.normalize(W_non,    dim=1)                 # (V, D)

    results_static = {}
    results_non    = {}

    for w in word_list:
        if w not in word2idx:
            if print_table:
                print(f"[nearest] '{w}' not in vocab")
            continue

        wid = word2idx[w]

        # Static 채널
        sims_s = torch.matmul(W_static, W_static[wid])      # (V,)
        sims_s[wid] = -float("inf")                         # 자기 자신 제외
        if 0 <= padding_idx < sims_s.numel():
            sims_s[padding_idx] = -float("inf")             # PAD 제외
        top_idx_s = torch.topk(sims_s, k=topk).indices.tolist()
        neigh_s = [idx2word[i] for i in top_idx_s]
        results_static[w] = neigh_s

        # Non-static 채널
        sims_n = torch.matmul(W_non, W_non[wid])            # (V,)
        sims_n[wid] = -float("inf")
        if 0 <= padding_idx < sims_n.numel():
            sims_n[padding_idx] = -float("inf")
        top_idx_n = torch.topk(sims_n, k=topk).indices.tolist()
        neigh_n = [idx2word[i] for i in top_idx_n]
        results_non[w] = neigh_n

    # --- 표 출력 (Kim2014 Table 3 스타일) ---
    if print_table:
        title_l = "Static Channel"
        title_r = "Non-static Channel"
        print("\nMost Similar Words (cosine) — multichannel")
        print(f"{title_l:<30} | {title_r:<30}")
        print("-" * 63)
        for w in word_list:
            if w not in results_static:   # vocab 밖이면 건너뜀
                continue
            L = results_static[w]
            R = results_non[w]
            print(f"\n{w}")
            m = max(len(L), len(R), 1)
            for i in range(m):
                ltok = L[i] if i < len(L) else ""
                rtok = R[i] if i < len(R) else ""
                print(f"  {ltok:<28} | {rtok:<28}")

    return {"static": results_static, "non_static": results_non}
        


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="CNN Sentence Classification")
    parser.add_argument("--model", type=str, choices=["rand", "static", "non-static", "multi"], default="rand")
    parser.add_argument("--task", type=str, choices=["cr", "mpqa", "mr", "sst1", "sst2", "subj", "trec"], default="mpqa")
    parser.add_argument("--seed", type=int, default=301)
    args = parser.parse_args()

    model_name, task, seed = args.model, args.task, args.seed

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    set_global_seed(args.seed)

    print(f"Selected MODEL: {model_name}")
    print(f"Selected TASK: {task}")

    # 1. 데이터 준비
    train_file = resolve_train_paths(task)
    print(train_file)
    dev_file   = resolve_dev_paths(task)
    print(dev_file)
    test_file = resolve_test_paths(task)
    print(test_file)

    sentences, labels, vocab = preprocess_data(train_file, task)
    n_labels = len(set(labels))
    print("[sentences]", len(sentences)) 
    print("[unique labels]", n_labels) 
    print("[vocab]", len(vocab))    

    word2idx, idx2word = build_vocab_mapping(vocab)
    vocab_size = len(word2idx)

    print("[word2idx size]", len(word2idx))
    print("[pad idx]", word2idx.get("<pad>", 0))
    print("[has '<pad>'?]", "<pad>" in word2idx)

    # 2. pretrained vector load
    pretrained = None
    if model_name != "rand":
        w2v = load_bin_vec("w2v.bin", set(vocab.keys()))
        print(len(w2v))
        add_oov_random(w2v, vocab.keys())
        print(len(w2v))
        pretrained = w2v


    # 3. 하이퍼파리미터
    hyperparams = {
        "dim": 300,
        "padding_idx": 0,
        "activation": "relu",
        "filter_sizes": [3, 4, 5],
        "num_feature_maps": 100,
        "p_dropout": 0.5,
        "l2_reg": 3,
        "batch_size": 50,
        "optimizer": "adadelta",
        "early_stop": "dev",
        "dev_ratio": 0.1           # dev set 없을 때 train의 10% 사용
    }
    print("[hyperparameters]")
    for k, v in hyperparams.items():
        print(f"  {k:<20}: {v}")


    # 4. 데이터셋별 학습
    proto = protocol(task, sentences, labels, dev_file, test_file, dev_ratio=hyperparams['dev_ratio'])
    
    results = []
    if proto["type"] == "cv":
        for foldinfo in proto["folds"]:
            k = foldinfo["fold"]
            tr_idx = foldinfo["train_idx"]
            dv_idx = foldinfo["dev_idx"]

            train_sents = [sentences[i] for i in tr_idx]
            train_labels = [labels[i] for i in tr_idx]
            dev_sents = [sentences[i] for i in dv_idx]
            dev_labels = [labels[i] for i in dv_idx]

            net = build_model(model_name, n_labels, word2idx, pretrained, hyperparams)
            dev_metric, _ = train(train_sents, train_labels, dev_sents, dev_labels, None, None,
                                net, hyperparams, word2idx)
            print(f"[CV fold {k}] dev_acc={dev_metric['acc']:.4f}")
            results.append(dev_metric["acc"])        
        if results:
            print(f"[CV] mean_dev_acc={np.mean(results) * 100:.1f} ± {np.std(results):.4f}")

    elif proto["type"] == "dev":
        train_sents = sentences
        train_labels = labels
        dev_sents, dev_labels   = proto["dev"]
        test_sents, test_labels = proto["test"]

        net = build_model(model_name, n_labels, word2idx, pretrained, hyperparams)
        dev_metric, test_metric = train(train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels,
                                        net, hyperparams, word2idx)
        print(f"[DEV] dev_acc={dev_metric['acc']:.4f} test_acc={test_metric['acc']:.4f}")

    elif proto["type"] == "trec":
        tr_ix = proto["train_idx"]
        dv_ix = proto["dev_idx"]

        train_sents = [sentences[i] for i in tr_ix]
        train_labels = [labels[i] for i in tr_ix]
        dev_sents    = [sentences[i] for i in dv_ix]
        dev_labels   = [labels[i] for i in dv_ix]
        test_sents, test_labels = proto["test"]

        net = build_model(model_name, n_labels, word2idx, pretrained, hyperparams)
        dev_metric, test_metric = train(train_sents, train_labels, dev_sents, dev_labels, test_sents, test_labels,
                                        net, hyperparams, word2idx)
        print(f"[TREC] dev_acc={dev_metric['acc']:.4f} test_acc={test_metric['acc']:.4f}")

    else:
        raise RuntimeError(f"Unknown protocol type: {proto['type']}")

    # 5. 정성평가
    target_words = ['bad', 'good', "n't", '!', ',']
    results = find_nearest_neighbors_multi(net, word2idx, idx2word, target_words, topk=10, padding_idx=hyperparams["padding_idx"], print_table=True)