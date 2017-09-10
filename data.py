import numpy as np
import json

from tqdm import tqdm


PADDING_TOKEN = '*PAD*'
UNK_TOKEN = '_'

default_vocab = {
    UNK_TOKEN: 0,
    PADDING_TOKEN: 1
}

LABEL_MAP = {
    "entailment": 0,
    "neutral": 1,
    "contradiction": 2,
}


class Example(object):
    pass        


def convert_binary_bracketing(parse, lowercase=False):
    transitions = []
    tokens = []

    for word in parse.split(' '):
        if word[0] != "(":
            if word == ")":
                transitions.append(1)
            else:
                # Lowercase all words to match GloVe.
                if lowercase:
                    tokens.append(word.lower())
                else:
                    tokens.append(word)
                transitions.append(0)
    return tokens, transitions


def incremental_tokenize(embedding_tokens, used_tokens, tokens):
    ret = []
    for token in tokens:
        if token in embedding_tokens and token is not UNK_TOKEN:
            if token not in used_tokens:
                used_tokens[token] = len(used_tokens)
            ret.append(token)
        else:
            ret.append(UNK_TOKEN)
    return ret


def preprocess_data(data_paths, embedding_path):
    """
    """

    # 1. Read embeddings to get candidate_dictionary.
    embedding_tokens = set()
    print("Reading embeddings for candidate tokens...")
    with open(embedding_path) as f:
        for i, line in tqdm(enumerate(f)):
            token = line[:line.find(' ')]
            embedding_tokens.add(token)

    print("# of candidate tokens: {}".format(len(embedding_tokens)))

    # 2. Read data to get real dictionary. And tokenize.
    datasets = []
    used_tokens = default_vocab.copy()
    for data_path in data_paths:
        dataset = []
        skipped = 0
        print("Reading data_path {}...".format(data_path))
        with open(data_path) as f:
            for i, line in tqdm(enumerate(f)):
                data = json.loads(line)
                ex = Example()

                label = data['gold_label']

                skip = False
                if label not in LABEL_MAP.keys():
                    skip = True

                if skip:
                    skipped += 1
                    continue

                ex.label = LABEL_MAP[label]
                ex.pairID = data['pairID']
                tokens1, _ = convert_binary_bracketing(data['sentence1_binary_parse'])
                tokens2, _ = convert_binary_bracketing(data['sentence2_binary_parse'])

                ex.tokens1 = incremental_tokenize(embedding_tokens, used_tokens, tokens1)
                ex.tokens2 = incremental_tokenize(embedding_tokens, used_tokens, tokens2)

                dataset.append(ex)

        print("# of examples: {}".format(len(dataset)))
        print("# of used tokens: {}".format(len(used_tokens)))
        print("skipped examples: {}".format(skipped))
        datasets.append(dataset)

    # 3. Read embeddings. Save in array.
    embedding_size = None
    with open(embedding_path) as f:
        for line in f:
            embedding_size = len(line.strip().split(' ')) - 1
            break

    print("embedding size: {}".format(embedding_size))

    vocab = default_vocab.copy()
    embeddings = []
    for _ in vocab.keys():
        embeddings.append([0.] * embedding_size)
    print("Reading embeddings to use...")
    with open(embedding_path) as f:
        for line in tqdm(f):
            token = line[:line.find(' ')]

            if token not in used_tokens:
                continue

            vocab[token] = len(vocab)
            vector = list(map(float, line.split(' ')[1:]))
            embeddings.append(vector)

    # 4. Tokenize again.
    print("Tokenizing...")
    for dataset in datasets:
        for ex in tqdm(dataset):
            ex.tokens1 = list(map(vocab.get, ex.tokens1))
            ex.tokens2 = list(map(vocab.get, ex.tokens2))

    embeddings = np.array(embeddings)
    print("embeddings shape {}:".format(embeddings.shape))

    return datasets, embeddings
