from collections import Counter


class Vocabulary:
    """Vocabulary for text."""

    def __init__(self, tokens=None, reserved_tokens=None, min_freq=2):
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []

        token_counts = count_corpus(tokens)
        self._token_freqs = sorted(
            token_counts.items(), key=lambda x: x[1], reverse=True)

        self.ndx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_ndx = {
            token: ndx for ndx, token in enumerate(self.ndx_to_token)
        }

        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_ndx:
                self.ndx_to_token.append(token)
                self.token_to_ndx[token] = len(self.ndx_to_token) - 1

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_ndx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def __len__(self):
        return len(self.ndx_to_token)

    def encode(self, seq):
        if isinstance(seq, str):
            seq = tokenize(seq)
            return self.__getitem__(seq)
        return self.__getitem__(seq)

        return ndxs

    def decode(self, ndxs: list):
        if not isinstance(ndxs, (list, tuple)):
            return self.ndx_to_token[ndxs]
        return ''.join((self.decode(ndx) + ' ') for ndx in ndxs)

    @property
    def unk(self):
        return 0

    @property
    def token_freqs(self):
        return self._token_freqs


def count_corpus(tokens: list):
    if len(tokens) == 0 or isinstance(tokens[0], (list, tuple)):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)


def tokenize(lines: list, method="word"):
    """Break sentences down into a list of tokens."""

    if method == "word":
        tokens = [line.split() for line in lines]
    elif method == "char":
        tokens = [list(line) for line in lines]
    elif method == "bpe":
        tokens = None   # Implement algorithm later
    else:
        raise ValueError(
            "Unknown method `{method}` specified. Must be either `word`, `char` or `bpe`.")

    return tokens
