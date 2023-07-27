import random
import os

import torch

from utils.tokenizer import Vocabulary, BPETokenizer, tokenize


# Helper functions to load data for BERT model

def read_data_for_bert(data_dir):
    file_name = os.path.join(data_dir, "dag-sents-train.txt")
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Convert all uppercase text into lower case
    paragraphs = [line.strip().lower().split(' . ')
                for line in lines if len(line.split(' . ')) >= 2]
    random.shuffle(paragraphs)
    return paragraphs


def get_tokens_and_segments(tokens_a, tokens_b=None):
    """Get tokens of BERT input sequences and their segment IDs."""

    tokens = ['<cls>'] + tokens_a + ['<sep>']
    segments = [0] * (len(tokens_a)+2)
    if tokens_b:
        tokens += tokens_b + ['<sep>']
        segments += [1] * (len(tokens_b)+1)
    return tokens, segments


def get_next_sentence(sentence, next_sentence, paragraphs):
    """Get next sentence and label (True/False) for nsp task,"""
    if random.random() < 0.5:
        is_next = True
    else:
        # `paragraphs` is a list of lists of lists
        next_sentence = random.choice(random.choice(paragraphs))
        is_next = False
    return sentence, next_sentence, is_next


def get_nsp_data_from_paragraph(paragraph, paragraphs, vocab, context_length):
    """Generate training examples for next sentece prediciton."""

    nsp_data_from_paragraph = []
    for i in range(len(paragraph)-1):
        tokens_a, tokens_b, is_next = get_next_sentence(
            paragraph, paragraph[i+1], paragraphs
        )
        # Consider 1 `<cls>` token and 2 `<sep>` tokens
        if len(tokens_a) + len(tokens_b) + 3 > context_length:
            tokens, segments = get_tokens_and_segments(tokens_a, tokens_b)
            nsp_data_from_paragraph.append((tokens, segments, is_next))
    return nsp_data_from_paragraph


def replace_mlm_tokens(tokens, candidate_pred_positions, num_mlm_preds, vocab):
    # Make a new copy of tokens for input to mlm
    mlm_input_tokens = [token for token in tokens]
    pred_positions_and_labels = []

    # Shuffle to get 15% random tokens for prediction in mlm
    for mlm_pred_position in candidate_pred_positions:
        if len(pred_positions_and_labels) >= num_mlm_preds:
            break
        masked_token = None
        # Replace word with mask 80% of the time
        if random.random() < 0.8:
            masked_token = "<MASK>"
        else:
            # Keep word unchanged 10% of the time
            if random.random() < 0.5:
                masked_token = tokens[mlm_pred_position]
            # Replace word with a random word 10% of the time
            else:
                masked_token = random.choice(vocab.ndx_to_token)
        mlm_input_tokens[mlm_pred_position] = masked_token
        pred_positions_and_labels.append(
            (mlm_pred_position, tokens[mlm_pred_position])
        )
    return mlm_input_tokens, pred_positions_and_labels


def get_mlm_data_from_tokens(tokens, vocab):
    candidate_pred_positions = []
    # tokens is a list of strings
    for i, token in enumerate(tokens):
        # We don't predict special (reserved) tokens in mlm task
        if token in ['<CLS>', '<SEP>']:
            continue
        candidate_pred_positions.append(i)
    # We predict 15% of tokens in language modeling task
    num_mlm_preds = max(1, round(len(tokens)*0.15))
    mlm_input_tokens, pred_positions_and_labels = replace_mlm_tokens(
        tokens, candidate_pred_positions, num_mlm_preds, vocab
    )
    pred_positions_and_labels = sorted(
        pred_positions_and_labels, key=lambda x: x[0]
    )
    pred_positions = [v[0] for v in pred_positions_and_labels]
    mlm_pred_labels = [v[1] for v in pred_positions_and_labels]
    return vocab[mlm_input_tokens], pred_positions, vocab[mlm_pred_labels]


def pad_bert_inputs(examples, context_length, vocab):
    max_num_mlm_preds = round(context_length * 0.15)
    all_token_ids, all_segments, attn_mask = [], [], []
    all_pred_positions, all_mlm_weights, all_mlm_labels = [], [], []
    nsp_labels = []
    for (token_ids, pred_positions, mlm_pred_label_ids, segments, is_next) in examples:
        all_token_ids.append(torch.tensor(token_ids + [vocab['<PAD>']] * (
            context_length - len(token_ids)), dtype=torch.long))
        all_segments.append(torch.tensor(segments + [0] * (
            context_length - len(segments)), dtype=torch.long))
        # attn_mask excludes count of <pad> tokens
        attn_mask.append(torch.tensor(len(token_ids), dtype=torch.float32))
        all_pred_positions.append(torch.tensor(pred_positions + [0] * (
            max_num_mlm_preds - len(pred_positions)), dtype=torch.long))
        # Prediction of <PAD> token will be excluded in the computation
        # of loss by multiplying by 0 weight
        all_mlm_weights.append(
            torch.tensor([1.0] * len(mlm_pred_label_ids) + [0.0] * (
                max_num_mlm_preds - len(pred_positions)), dtype=torch.float32))
        all_mlm_labels.append(torch.tensor(mlm_pred_label_ids + [0] * (
            max_num_mlm_preds - len(mlm_pred_label_ids)), dtype=torch.long))
        nsp_labels.append(torch.tensor(is_next, dtype=torch.long))
        return (all_token_ids, all_segments, attn_mask, all_pred_positions,
                all_mlm_weights, all_mlm_labels, nsp_labels)


class DatasetForBertModel(torch.utils.data.Dataset):
    def __init__(self, paragraphs, context_length):
        # pargraphs[i] is a list of sentence strings that represent a paragraph
        # where each sentence is a list of tokens
        paragraphs = [tokenize(paragraph, method="word")
                      for paragraph in paragraphs]
        sentences = [
            sentence for paragraph in paragraphs for sentence in paragraph]
        self.vocab = Vocabulary(sentences, min_freq=5, reserved_tokens=[
                                '<PAD>', '<MASK>', '<CLS>', '<SEP>'])

        # Get data for nsp task
        examples = []
        for paragraph in paragraphs:
            examples.extend(get_nsp_data_from_paragraph(
                paragraph, paragraphs, self.vocab, context_length
            ))

        # Get data for mlm task
        examples = [(get_mlm_data_from_tokens(tokens, self.vocab)
                     + (segments, is_next)) for tokens, segments, is_next in examples]

        # Pad input to uniform length
        (self.all_token_ids, self.all_segments, self.attn_mask,
         self.all_pred_positions, self.all_mlm_weights,
         self.all_mlm_labels, self.all_nsp_labels) = pad_bert_inputs(
            examples, context_length, self.vocab
        )

    def __getitem__(self, ndx):
        return (
            self.all_token_ids[ndx], self.all_segments[ndx],
            self.attn_mask[ndx], self.all_pred_positions[ndx],
            self.all_mlm_weights[ndx], self.all_mlm_labels[ndx],
            self.nsp_labels[ndx]

        )

    def __len__(self):
        return len(self.all_token_ids)


def get_data_iter_for_bert(data_dir, batch_size, context_length):
    paragraphs = read_data_for_bert(data_dir)
    train_set = DatasetForBertModel(paragraphs, context_length)
    train_iter = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                             shuffle=True)
    return train_iter, train_set.vocab


# Helper functions to load data for GPT model

def read_data_for_gpt(data_dir, in_lower=False):
    filename = os.path.join(data_dir, "dag-sents-train.txt")
    with open(filename, "r", encoding="utf-8") as f:
        text = f.read()
    if in_lower:
        return text.lower().strip()
    return text.strip()


def encode_sequence_for_gpt(text):
    # Fetch pretrained tokenizer
    dagpt_tokenizer = BPETokenizer.from_pretrained("configs/dagpt-base-uncased-tokenizer.json")
    enc_text = dagpt_tokenizer.encode(text)
    return enc_text

def get_gpt_batch(enc_text, context_length, batch_size, device_type="cuda"):
    """
    Generate a set of input and target tokens. Stacked together into
    `batch_size` examples at a time.
        enc_text: token indices of tokenized text.
        context_length: context length of gpt model.
        batch_size: batch size.
    """
    ndxs = torch.randint(len(enc_text)-context_length, (batch_size,))
    xs = torch.stack([torch.tensor(enc_text[ndx:ndx+context_length], dtype=torch.int64) for ndx in ndxs])
    ys = torch.stack([torch.tensor(enc_text[ndx+1:ndx+1+context_length], dtype=torch.int64) for ndx in ndxs])
    if device_type == "cuda":
        xs, ys = xs.pin_memory().to(torch.device("cuda"), non_blocking=True), ys.pin_memory().to(torch.device("cuda"), non_blocking=True)
    else:
        xs, ys = xs.to(torch.device("cpu")), ys.to(torch.device("cpu"))
    return (xs, ys)
