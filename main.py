import logging

from MyTransformer import *


def one_of_top(distribution, k=3):
    k_probs, k_indices = torch.topk(distribution, k, dim=-1)
    sampled_id_in_k_indices = torch.multinomial(k_probs, num_samples=1)
    one_in_top_k = torch.gather(k_indices, dim=-1, index=sampled_id_in_k_indices)
    return one_in_top_k


def model_predict(input_sentences, *, vocab, model, device, max_output_len=128):
    model.load_state_dict(torch.load("./weight.pth", map_location="cpu"))
    model.eval()
    model = model.to(device)

    x_dec = vocab.sentence2mtx([""], bos=True).to(device)
    x_enc = vocab.sentence2mtx(input_sentences, eos=True).to(device)
    for i in range(max_output_len):
        model_output, _ = model(x_enc, x_dec)
        word_distribution = model_output[0, 0]
        next_token_id = one_of_top(word_distribution, k=3)
        next_token = torch.tensor([[next_token_id]]).to(device)
        x_dec = torch.cat([x_dec, next_token], dim=-1)
        if next_token_id.item() == vocab.eos_id:
            break
    generated_words = vocab.idx2word(x_dec[0].tolist())
    return generated_words


def main(debug=False):   # Set this to True to output debugging info
    log.debug_mode(debug)

    word_list = model_predict(["I would like to know if it is a good idea to go out tomorrow."],
                              vocab=VOCABULARY, model=MODEL, device=DEVICE)
    print("Word List: ", word_list)
    generated_sentence = " ".join(word_list[1:-2]) + word_list[-2]
    print(generated_sentence)


if __name__ == "__main__":
    main()
