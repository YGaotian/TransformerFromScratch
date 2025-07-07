from MyTransformer import *
import torch.optim as optim


def train(model, vocab, device, num_epoch=50, learning_rate=0.001):
    # Data configuration
    data_A = [
        "Hello, world!",
        "I am doing well.",
        "Sure, my name is Tom.",
        "Nice to meet you, too.",
        "I want to know the weather outside.",
        "Ah, nothing else, but do you have time tonight?",
        "Got it, thank you!"
    ]
    data_B = [
        "Hey, how are you doing?",
        "Great! May I have your name, sir?",
        "Nice to meet you, Tom!",
        "So, how can I help you today?",
        "It is sunny. How beautiful it is!",
        "Sure, I will make a phone call to you when I am off duty.",
        "You are welcome! Anything else I can do for you?"
    ]
    input_enc = vocab.sentence2mtx(data_A, eos=True).to(device)
    input_dec = vocab.sentence2mtx(data_B, bos=True).to(device)
    target_lbl = vocab.sentence2mtx(data_B, eos=True).to(device)

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
        if (epoch + 1) % 10 == 0:
            print("Epoch " + str(epoch) + " loss: " + str(loss.item()))

    total_loss /= len(input_enc)
    print("total loss: " + str(total_loss))
    torch.save(model.state_dict(), "weight.pth")

log.debug_mode(False)  # Set this to True for debugging information
train(model=MODEL, vocab=VOCABULARY, device=DEVICE, num_epoch=700)
