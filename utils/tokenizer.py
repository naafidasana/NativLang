import json
import string

from collections import Counter, defaultdict
import os


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

    def encode(self, seq: list):
        return self.__getitem__(seq)


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


def tokenize(lines: list, method="word"):
    """Break sentences down into a list of tokens."""
    if not isinstance(lines, (list, tuple)):
        lines = [lines]

    if method == "word":
        tokens = [line.lower().split() for line in lines]
    elif method == "char":
        tokens = [list(line) for line in lines]
    else:
        raise ValueError(
            "Unknown method `{method}` specified. Must be either `word` or `char`.")

    return tokens


def count_corpus(tokens: list):
    if len(tokens) == 0 or isinstance(tokens[0], (list, tuple)):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return Counter(tokens)


class BPETokenizer:

    def __init__(self, vocab_size=30272):
        self.vocab_size = vocab_size
        self.vocab = None
        self.inv_vocab = None

    def train(self, text):
        """Train tokenizer on text."""
        self.words = self._preprocess(text)
        word_freqs = Counter(self.words)
        self.vocab = self._build_vocab(self.vocab_size, word_freqs)
        self.inv_vocab = {int(ndx): token for token, ndx in self.vocab.items()}

    def _preprocess(self, text):

        text = text.lower()
        tokens = text.split()
        tokens = [' '.join(token) + ' </w>' for token in tokens]

        return tokens

    def _get_pairs(self, word_freqs):
        pairs = Counter()
        for word, freq in word_freqs.items():
            chars = word.split()  # Split the word into individual characters
            char_zip = zip(chars, chars[1:])  # Pair adjacent characters using a sliding window
            pairs.update(char_zip)  # Update the pairs Counter with the new pairs

        return pairs

    def _merge_pairs(self, best_pair, word_freqs):
        best_pair_space = ' '.join(best_pair)
        best_pair_merged = "".join(best_pair)

        # Create a new Counter to store the merged word frequencies
        merged_dict = Counter()
        for word, freq in word_freqs.items():
            merged_word = word.replace(best_pair_space, best_pair_merged)
            merged_dict[merged_word] += freq

        return merged_dict

    def _get_subword_tokens(self, word_freqs):
        """Get subword tokens."""
        char_counts = defaultdict(int)
        for word, freq in word_freqs.items():
            chars = word.split()
            for char in chars:
                char_counts[char] += freq

        return char_counts

    def _build_vocab(self, n_iter, word_freqs):
        # Merge all subword tokens from all iterations into a single Counter
        subword_tokens = Counter()
        for i in range(n_iter-1):   # UNK token will be added later
            pairs = self._get_pairs(word_freqs)
            best_pair = max(pairs, key=pairs.get)
            word_freqs = self._merge_pairs(best_pair, word_freqs)
            subword_tokens.update(self._get_subword_tokens(word_freqs))

        # Sort subword_tokens based on frequency in descending order
        sorted_tokens = sorted(subword_tokens.items(), key=lambda x: x[1], reverse=True)

        # Select top self.vocab_size tokens to build the vocabulary
        vocab = {token: idx for idx, (token, _) in enumerate(sorted_tokens[:self.vocab_size])}

        # Add the special <unk> token to the vocabulary
        vocab['<unk>'] = len(vocab)
        return vocab


    def encode(self, text):
        """Encode a string into token id's."""
        if self.vocab is None:
            raise ValueError("No tokenizer has been trained yet.")
        tokens = self._preprocess(text)
        encoded_tokens = []

        for token in tokens:
            merged_token = ""  # Initialize the merged token
            prev_index = None  # Store the previous token index
            for char in token.split():  # Split the token into individual characters
                merged_token += char  # Append the character to the merged_token
                if merged_token in self.vocab:
                    prev_index = self.vocab[merged_token]  # Get the token index
                else:
                    # If the merged_token is not in the vocabulary, it's part of the <unk> token
                    # Use the previous token's index
                    encoded_tokens.append(prev_index)
                    merged_token = char  # Continue with the rest of the token

            # If we reach the end of a token (</w>), add its index to the encoded tokens
            if "</w>" in token:
                encoded_tokens.append(prev_index)

        return encoded_tokens
    
    def decode(self, token_ndxs):
        """Decode a sequence of token indices into text."""
        if self.vocab is None:
            raise ValueError("No tokenizer has been trained yet.")
        tokens = [self.inv_vocab[ndx] if ndx in self.inv_vocab else "<unk>" for ndx in token_ndxs]
        return " ".join(tokens).replace("</w>", '')

    def save(self, save_path):
        tokenizer_state = {
            "vocab_size": self.vocab_size,
            "vocab": self.vocab,
            "inv_vocab": self.inv_vocab
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(tokenizer_state, f, ensure_ascii=False)

    @classmethod
    def from_pretrained(cls, save_path):
        """Load pretrained tokenizer."""
        with open(save_path, "r", encoding="utf-8") as f:
            tokenizer_state = json.load(f)

        tokenizer = cls(tokenizer_state["vocab_size"])
        tokenizer.vocab = tokenizer_state["vocab"]
        tokenizer.inv_vocab = tokenizer_state["inv_vocab"]
        tokenizer.inv_vocab = {int(ndx):token for ndx, token in tokenizer.inv_vocab.items()}

        return tokenizer
