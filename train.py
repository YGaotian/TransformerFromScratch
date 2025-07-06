from MyTransformer import *
import torch.optim as optim


def train(num_epoch=10, learning_rate=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    vocab = Vocabulary()
    vocab_size = vocab.getVocabSize()
    # Data configuration
    data_A = [
        "Hello, world!",
        "I am doing well.",
        "Sure, my name is Tom.",
        "Nice to meet you, too.",
        "I want to know the weather outside."
    ]
    data_B = [
        "Hey, how are you doing?",
        "Great! May I have your name, sir?",
        "Nice to meet you, Tom!",
        "So, how can I help you today?",
        "It is sunny. How beautiful it is!"
    ]
    input_enc = torch.tensor(vocab.sentence2mtx(data_A, eos=True)).to(device)
    input_dec = torch.tensor(vocab.sentence2mtx(data_B, bos=True)).to(device)
    target_lbl = torch.tensor(vocab.sentence2mtx(data_B, eos=True)).to(device)

    log(input_dec)
    log(target_lbl)

    model = Transformer(32, 4, 8, vocab_size,
                        0.2, 128, 4).to(device)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    model.train()
    total_loss = 0
    for epoch in range(num_epoch):
        optimizer.zero_grad()
        output, loss = model(input_enc, input_dec, target_lbl)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 2 == 0:
            print(loss.item())

    total_loss /= len(input_enc)
    print(total_loss)
    torch.save(model.state_dict(), "weight.pth")

log.debug_mode(True)  # Set this to True for debugging information
train()
