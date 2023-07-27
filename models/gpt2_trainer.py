from models.gpt2_model import GPTModel, GPTConfig
from utils.general_utils import Visualizer, MetricAccumulator, Timer, get_gpus, download_and_extract
from utils.dataloaders import get_gpt_batch, read_data_for_gpt
from utils.tokenizer import BPETokenizer

import torch
import torch.nn as nn

import os

# Download (or fetched cached dataset) and obtain path to dataset
DATA_DIR = r".\data\dag-sents-train"
text = read_data_for_gpt(DATA_DIR)

# Get pretreained tokenizer
tokenizer = BPETokenizer.from_pretrained("configs/dagpt-base-uncased-tokenizer.json")
vocab_size = tokenizer.vocab_size

# Encode text
enc_text = tokenizer.encode(text)

# Get data torch data loaders and vocabulary
context_length = 128
batch_size = 16

# Create model config and model
config = GPTConfig(vocab_size, context_length)
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


def train_gpt(config, model, train_data, learning_rate, num_steps):
    
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
    step, timer = 0, Timer()
    visualizer = Visualizer(xlabel="epoch", ylabel="loss",
                            xlim=[1, num_steps])
    metrics = MetricAccumulator(3)

    print("Training...")
    while step < num_steps:
        # Get batch inputs and targets
        batch_xs, batch_ys = get_gpt_batch(train_data, config.context_length, batch_size)
        
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
        visualizer.add(step + 1, metrics[0]/metrics[2])

    step += 1

    # Save model after every 300 steps
    # We process `batch_size` number of examples in each step
    if (step % 300 == 0):
        checkpoint_name = os.path.join(checkpoint_path, f"dagpt-{step:03}.pth")
        # Save model
        torch.save(model.module.state_dict(), checkpoint_name)

        # Print generated sequence
        try_generate(model)

        print(f"Loss: {metrics[0]/metrics[2]:.4f}")
        print(f"{metrics[1]/timer.sum():.1f} tokens/sec on {str(devices)}")


def try_generate(model, max_tokens=10):
    seq = "Di nyɛla bikura shikuru shɛli"
    # Encode sequence
    encoded_seq = tokenizer.encode(seq)
    encoded_seq = torch.tensor(encoded_seq).to(devices[0])
    with torch.no_grad():
        try:
            gen_seq = model.module.generate(encoded_seq, max_tokens)
        except:
            gen_seq = model.generate(encoded_seq, max_tokens)
        gen_seq = gen_seq.cpu().numpy().tolist()[0]
        # Decode generated sequence
        print(tokenizer.decode(gen_seq))


print(f"Number of Parameters: {sum([p.numel() for p in model.parameters()])/1e6} M")
