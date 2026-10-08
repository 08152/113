import re


class SimpleTokenizer:
    def __init__(self):
        self.vocab = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3
        }

        self.reverse_vocab = {
            0: "<PAD>",
            1: "<UNK>",
            2: "<BOS>",
            3: "<EOS>"
        }

    def tokenize(self, text):
        text = text.lower()

        tokens = re.findall(
            r"\w+|[^\w\s]",
            text,
            re.UNICODE
        )

        return tokens

    def add_tokens(self, tokens):
        for token in tokens:
            if token not in self.vocab:
                token_id = len(self.vocab)

                self.vocab[token] = token_id
                self.reverse_vocab[token_id] = token

    def encode(self, text):
        tokens = self.tokenize(text)

        ids = []

        for token in tokens:
            if token in self.vocab:
                ids.append(self.vocab[token])
            else:
                ids.append(self.vocab["<UNK>"])

        return ids

    def decode(self, ids):
        tokens = []

        for token_id in ids:
            token = self.reverse_vocab.get(
                token_id,
                "<UNK>"
            )

            if token not in ["<PAD>", "<BOS>", "<EOS>"]:
                tokens.append(token)

        text = ""

        for token in tokens:
            if token in ".,!?;:":
                text += token
            else:
                if text:
                    text += " "

                text += token

        return text

    def train(self, texts):
        for text in texts:
            tokens = self.tokenize(text)
            self.add_tokens(tokens)

    def vocab_size(self):
        return len(self.vocab)
