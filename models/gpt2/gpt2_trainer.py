from models.gpt2.gpt2_model import GPTModel, GPTConfig
from utils.general_utils import Visualizer, MetricAccumulator, Timer, get_gpus, download_and_extract
from utils.dataloaders import get_data_iter_for_gpt
from utils.tokenizer import tokenize

import torch
import torch.nn as nn

import os

# Download (or fetched cached dataset) and obtain path to dataset
DATA_DIR = "./data/dag-sents-train"

# Get data torch data loaders and vocabulary
context_length = 12
batch_size = 256
train_iter, vocab = get_data_iter_for_gpt(DATA_DIR, context_length, batch_size)

# Create model config and model
config = GPTConfig(len(vocab), context_length)
model = GPTModel(config)

devices, num_devices = get_gpus()


def grad_clipping(model, theta):
    params = [p for p in model.parameters() if p.requires_grad()]
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2) for p in params)))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm


def get_gpt_batch_loss(model, input_sequences, targets):
    # Loss is also calculated in the forward pass of the model
    logits, loss = model(input_sequences, targets)
    return loss.sum()


def train_gpt(model, train_iter, learning_rate, num_epochs):
    
    # Add code to enable checkpointing
    checkpoint_path = "./checkpoints"
    if not os.path.exists(checkpoint_path):
        os.mkdir(checkpoint_path)

    # Get and move model to device.
    if num_devices > 0:
        # Use parallel processing on multiple GPUs.
        model = nn.DataParallel(model, device_ids=devices).to(devices[0])
    else:
        # Still explicitly move model to device incase there is only one GPU.
        model.to(devices[0])
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    epoch, timer = 0, Timer()
    visualizer = Visualizer(xlabel="epoch", ylabel="loss",
                            xlim=[1, num_epochs])
    metrics = MetricAccumulator(3)

    print("Training...")
    while epoch < num_epochs:
        # Get batch inputs and targets
        for batch_xs, batch_ys in train_iter:
            batch_xs, batch_ys = batch_xs.to(
                devices[0]), batch_ys.to(devices[0])

            # Reset any previously computed gradients
            optimizer.zero_grad()

            # Do forward pass and fetch loss
            timer.start()
            loss = get_gpt_batch_loss(model, batch_xs, batch_ys)

            # Backward pass
            loss.backward()
            #grad_clipping(model, 1)
            optimizer.step()
            metrics.add(loss,
                        batch_xs.shape[0], 1)
            timer.stop()
            visualizer.add(epoch + 1, metrics[0]/metrics[2])

        epoch += 1

        # Save model after every two epochs
        if (epoch % 2 == 0):
            checkpoint_name = os.path.join(checkpoint_path, f"dagpt-{epoch:03}.pth")
            # Save model
            torch.save(model.module.state_dict(), checkpoint_name)

        print(f"Loss: {metrics[0]/metrics[2]:.4f}")
        print(f"{metrics[1]/timer.sum():.1f} tokens/sec on {str(devices)}")


# Train model in notebook
def run_training(learning_rate=1e-4, num_epochs=10):
    train_gpt(model, train_iter, learning_rate=learning_rate, num_epochs=num_epochs)
