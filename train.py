import torch.optim as optim
from torch.utils.data import DataLoader
from DataProcess import *
from WarmupCosineAnnealing import warmup_cosine_annealing
from setup import *
from functools import partial
import os


def train(
        model,
        device,
        data_path,
        batch_size,
        checkpoint_dir,
        incremental_save=False,
        num_epoch=40,
        learning_rate=0.0001,
        continue_from=None
):
    # Load Data
    dialog_dataset = DialogDataset(data_path, tell=True)
    packer_wrapper = partial(packer, vocab=VOCABULARY, device=DEVICE)
    data_loader = DataLoader(dialog_dataset,
                             batch_size,
                             shuffle=True,
                             num_workers=0,
                             collate_fn=packer_wrapper)

    print(f"Training on: {device}.")

    # Load a checkpoint as needed
    checkpoint = None
    if continue_from and os.path.exists(continue_from):
        checkpoint = torch.load(continue_from, map_location="cpu")
        print(f"Training starts from the file: {continue_from}.")
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        print("Training starts from scratch.")

    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
    # Set up cosine annealing with warmup
    batch_num_per_epoch = len(data_loader)
    all_training_steps = batch_num_per_epoch * num_epoch
    warmup_steps = int(0.05 * all_training_steps)
    scheduler = warmup_cosine_annealing(optimizer, warmup_steps, all_training_steps)

    start_epoch = 0

    # Resume checkpoint states
    if checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        start_epoch = checkpoint["epoch"]
        print(f"Last epoch loss: {checkpoint['loss']}")

    model.train()

    for epoch in range(start_epoch, num_epoch):
        total_loss = 0
        for batch_id, (input_enc, input_dec, target_lbl) in enumerate(data_loader):
            optimizer.zero_grad()
            output, loss = model(input_enc, input_dec, target_lbl)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch}, Batch {batch_id}/{len(data_loader) - 1}, "
                  f"Loss: {loss.item():.4f}, lr: {optimizer.param_groups[0]['lr']:.8f}")
            scheduler.step()

        avg_loss = total_loss / len(data_loader)
        print(f"======Average loss per batch: {avg_loss:.4f}======")
        training_id = "only"
        if incremental_save:
            ones = int(avg_loss // 1)
            pts = int(10000 * (avg_loss % 1))
            training_id = f"E{epoch}L{ones}_{pts}"
        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "loss": avg_loss}, f"{checkpoint_dir}/checkpoint_{training_id}.pth")

        print(f"Checkpoint is saved for epoch {epoch}")


if __name__ == "__main__":
    log.debug_mode(False)  # Set this to True for debugging information

    BATCH_SIZE = 128
    DATA = "./Data/dialog_list.json"

    train(model=MODEL,
          device=DEVICE,
          data_path=DATA,
          batch_size=BATCH_SIZE,
          checkpoint_dir="./Saved_Params",
          incremental_save=True,
          num_epoch=60,
          learning_rate=0.0008)
