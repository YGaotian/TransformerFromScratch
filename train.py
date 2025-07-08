import torch.optim as optim
from DataGetter import *
from torch.utils.data import DataLoader
from setup import *
from functools import partial


packer_wrapper = partial(packer, vocab=VOCABULARY, device=DEVICE)

loader = DataLoader(dialog_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, collate_fn=packer_wrapper)


def train(data_loader, model, device, num_epoch=100, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()
    total_loss = 0

    for epoch in range(num_epoch):
        for batch_id, (input_enc, input_dec, target_lbl) in enumerate(data_loader):
            optimizer.zero_grad()
            output, loss = model(input_enc, input_dec, target_lbl)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
            print(f"Epoch {epoch + 1}, Batch {batch_id + 1}/{len(data_loader)}, Loss: {loss.item():.4f}")

    total_loss /= len(data_loader.dataset)
    print("total loss: " + str(total_loss))
    torch.save(model.state_dict(), "weight.pth")

if __name__ == "__main__":
    log.debug_mode(False)  # Set this to True for debugging information
    train(loader, model=MODEL, device=DEVICE, num_epoch=100)
