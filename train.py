from MyTransformer import *
import torch.optim as optim
from dataset import *


def train(enc_data, dec_data, model, vocab, device, num_epoch=50, learning_rate=0.001):
    input_enc = vocab.sentence2mtx(enc_data, eos=True).to(device)
    input_dec = vocab.sentence2mtx(dec_data, bos=True).to(device)
    target_lbl = vocab.sentence2mtx(dec_data, eos=True).to(device)

    log(input_dec)
    log(target_lbl)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.to(device)
    model.train()
    total_loss = 0
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        output, loss = model(input_enc, input_dec, target_lbl)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 50 == 0:
            print("Epoch " + str(epoch + 1) + " loss: " + str(loss.item()))

    total_loss /= len(input_enc)
    print("total loss: " + str(total_loss))
    torch.save(model.state_dict(), "weight.pth")

log.debug_mode(False)  # Set this to True for debugging information
train(enc_data=data_A, dec_data=data_B, model=MODEL, vocab=VOCABULARY, device=DEVICE, num_epoch=1500)
