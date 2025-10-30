import os
import re

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


def get_cr(path, task):
    labels = []
    sentences = []
    product_ls = os.listdir(path)
    positive_negative_pattern = r'(\+?\-?)\d+'

    for product in product_ls:
        data_path = path + product
        with open(data_path, "rb") as file:
            document = file.readlines()

            for sentence in document:
                sentence = sentence.decode(errors="replace")
                if "##" in sentence:
                    label_sentence = sentence.split("##")
                    if len(label_sentence) != 2:
                        continue
                    if label_sentence[0] != '':
                        processed_sentence = clean_str(label_sentence[1], task=task)
                        processed_sentence = processed_sentence.strip().split(" ")
                        processed_sentence = [token.strip() for token in processed_sentence]

                        category_score = re.findall(positive_negative_pattern, label_sentence[0])
                        category_score = set(category_score)
                        if set("+") == category_score or set(["+",""]) == category_score:
                            labels.append("positive")
                            sentences.append(processed_sentence)
                        elif set("-") == category_score or set(["-",""]) == category_score:
                            labels.append("negative")
                            sentences.append(processed_sentence)
                        else:
                            continue
    
    return sentences, labels


def save_cr_to_file(path, task, output_path="data/cr/cr_all.txt"):
    sentences, labels = get_cr(path, task)

    with open(output_path, "w", encoding="utf-8") as f:
        for label, tokens in zip(labels, sentences):
            if label == 'negative':
                line = "0" + " " + " ".join(tokens) + "\n"
                f.write(line)
            elif label == 'positive':
                line = "1" + " " + " ".join(tokens) + "\n"
                f.write(line)

    print(f"[+] Saved {len(sentences)} samples to {output_path}")


get_cr('data/cr/', 'cr')
save_cr_to_file('data/cr/', 'cr', output_path="data/cr/cr_all.txt")