from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
)
import torch

def simulate():
    tokenizer = T5Tokenizer.from_pretrained("t5-base")
    model = T5ForConditionalGeneration.from_pretrained("model")

    prefix = "translate Indonesian to English: "
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    torch = model.to(device)

    while True:
        text = input("Input Text: ")
        input_ = prefix + text + " </s>"
        print("Input_: ", input_)
        input_ = tokenizer.encode(text, return_tensors='pt').to(device)
        out = model.generate(input_)
        out = tokenizer.decode(out[0])
        print("Translated: ", out)
