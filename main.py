from setup import *
from MyTransformer import *
import os


def one_of_top(distribution, k=3):
    assert k >= 1, "Cannot select less than 1 option."
    k_probs, k_indices = torch.topk(distribution, k, dim=-1)
    sampled_id_in_k_indices = torch.multinomial(k_probs, num_samples=1)
    one_in_top_k = torch.gather(k_indices, dim=-1, index=sampled_id_in_k_indices)
    return one_in_top_k


def model_predict(input_sentences, *, vocab, model, device, param_file, freedom_degree=3, max_output_len=128):
    if not os.path.exists(param_file):
        print("Cannot find parameters.")
    param_dict = torch.load(param_file, map_location="cpu")
    model.load_state_dict(param_dict["model_state_dict"])
    model.eval()
    model = model.to(device)

    x_dec = vocab.sentence2mtx([""], bos=True).to(device)
    x_enc = vocab.sentence2mtx(input_sentences, eos=True).to(device)
    for i in range(max_output_len):
        model_output, _ = model(x_enc, x_dec)
        word_distribution = model_output[0, 0]
        next_token_id = one_of_top(word_distribution, k=freedom_degree)
        next_token = torch.tensor([[next_token_id]]).to(device)
        x_dec = torch.cat([x_dec, next_token], dim=-1)
        if next_token_id.item() == vocab.eos_id:
            break
    generated_words = vocab.idx2word(x_dec[0].tolist())
    return generated_words


def main(debug=False):   # Set this to True to output debugging info
    log.debug_mode(debug)

    input_sentence = "I am going to train a model."
    word_list = model_predict([input_sentence],
                              vocab=VOCABULARY,
                              model=MODEL,
                              device=DEVICE,
                              param_file="./Saved_Params/checkpoint_E59L1_4517.pth",
                              freedom_degree=2)

    generated_sentence = " ".join(word_list[1:-2]) + word_list[-2]
    print("All model parameters: " + str(MODEL.num_of_param(count_embedding=False)))
    print("You say: " + input_sentence)
    print("AI answers: " + generated_sentence)


if __name__ == "__main__":
    main()
