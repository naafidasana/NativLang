from bert_model import BERTModel, BertConfig
from utils.general_utils import Visualizer, MetricAccumulator, Timer, get_gpus, download_and_extract
from utils.dataloaders import get_data_iter_for_bert

import torch
import torch.nn as nn


#DATA_DIR = download_and_extract("dag-wiki-data")
DATA_DIR = r".\data\dag-sents-train"

# Get torch data loaders and vocabulary
context_length, batch_size = 64, 4
train_iter, vocab = get_data_iter_for_bert(DATA_DIR, batch_size, context_length)

# Create BERT config and BERT model
vocab_size = len(vocab)
config = BertConfig(vocab_size, context_length)
model = BERTModel(config)

criterion = nn.CrossEntropyLoss()
devices, num_devices = get_gpus()


def get_bert_batch_loss(model, criterion, vocab_size, tokens_X,
                        segments_X, attn_mask, pred_positions_X,
                        mlm_weights_X, mlm_Y, nsp_Y):
    # Forward pass
    _, mlm_Y_hat, nsp_Y_hat = model(tokens_X, segments_X,
                                    attn_mask.reshape(-1), pred_positions_X)

    # Caulate mlm loss
    mlm_loss = criterion(mlm_Y_hat.reshape(-1, vocab_size), mlm_Y.reshape(-1)) *\
        mlm_weights_X.reshape(-1, 1)
    mlm_loss = mlm_loss.sum() / (mlm_weights_X.sum() + 1e-8)

    # Calculate nsp loss
    nsp_loss = criterion(nsp_Y_hat, nsp_Y)
    total_loss = mlm_loss + nsp_loss
    return mlm_loss, nsp_loss, total_loss


def train_bert_model(model, train_iter, criterion, lr, num_epochs):
    if num_devices > 1:
        # There are multiple devices. Do parallel processing
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    else:
        # Still explicitly model to `devices` in case there is only one GPU on the system.
        model.to(devices[0])

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    epoch, timer = 0, Timer()
    visualizer = Visualizer(xlabel="epoch", ylabel="loss",
                            xlim=[1, num_epochs], legend=['mlm', 'nsp'])
    # Store sum of mlm losses, sum of nsp losses, no. of sentence pairs,
    # count in an accumulator for metrics (from the `MetricAccumulator` class)
    metrics = MetricAccumulator(4)
    print("Training...")
    while epoch < num_epochs:
        # Process process input one batch at a time
        for tokens_X, segments_X, attn_mask_X, pred_positions_X,\
                mlm_weights_X, mlm_Y, nsp_Y in train_iter:
            tokens_X = tokens_X.to(devices[0])
            segments_X = segments_X.to(devices[0])
            attn_mask_X = attn_mask_X.to(devices[0])
            pred_positions_X = pred_positions_X.to(devices[0])
            mlm_weights_X = mlm_weights_X.to(devices[0])
            mlm_Y, nsp_Y = mlm_Y.to(devices[0]), nsp_Y.to(devices[0])

            # Reset any previously computed gradients
            optimizer.zero_grad()

            # Do forward pass and compute loss
            timer.start()
            mlm_loss, nsp_loss, batch_loss = get_bert_batch_loss(
                model, criterion, vocab_size, tokens_X, segments_X, attn_mask_X,
                pred_positions_X, mlm_weights_X, mlm_Y, nsp_Y)

            # Backward pass
            batch_loss.backward()
            optimizer.step()
            metrics.add(mlm_loss, nsp_loss, tokens_X.shape[0], 1)
            timer.stop()
            visualizer.add(epoch + 1,
                           (metrics[0]/metrics[3], metrics[1]/metrics[3]))

        epoch += 1

        print(f"MLM Loss: {metrics[0]/metrics[3]:.4f} \t"
              f"NSP Loss: {metrics[1]/metrics[3]:.4f}")
        print(f"{metrics[2]/timer.sum():.1f} sentence pairs/sec on "
              f"{str(devices)}")


# Train model
lr, num_epochs = 1e-3, 10
train_bert_model(model, train_iter, criterion, lr, num_epochs)
