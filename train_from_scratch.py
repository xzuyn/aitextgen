from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer
from aitextgen.utils import GPT2ConfigCPU
from aitextgen import aitextgen
from math import floor
import logging
import wandb
import os
from chunk import token_chunk_split


batch_size = 1  # high batch size gives worse training(?), but is used to speed up training time
max_length = 1024  # default is 64
epochs = 4  # default is 1
save_every = 1000  # default is 5000
generate_every = 500  # default is 500
vocab_size = 8192  # default is (1024 * 1)
n_embed = 512  # default is 128
n_layer = 8  # default is 4
n_head = 8  # default is 4
line_by_line = False  # default is False
learning_rate = 0.001  # default is 0.001
dropout = 0.0  # default is 0.0
split_string = "<|thread|>"

# Trying to train a tokenizer with your <|endoftext|> token in the dataset
# will cause it to break it into multiple tokens, while also being a special token.
# So we train the tokenizer on a dataset without it to make sure it stays as a single special token.
# We will train the actual model on the no_bos version.
#
# file_name_with_bos = Your dataset WITH <|endoftext|> tokens.
# file_name_no_bos = Your dataset WITHOUT <|endoftext|> tokens.
file_name_with_bos = "./trained_model/dataset/filtered_chunks.txt"
file_name_no_bos = "./trained_model/dataset/filtered_chunks-nobos.txt" # TODO: automatically make this
wandb_project_name = "Pol"
wandb_run_name = "Pol-v2"  # Dial-EPOCH-1


# Your code needs to be wrapped inside a main function,
# as otherwise multiple child processes from pytorch_lightning cannot be spawned
def main():
    # Setup logging
    global reload
    logging.basicConfig(
        format="%(asctime)s — %(levelname)s — %(name)s — %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO
    )

    if not os.path.exists("./trained_model/pytorch_model.bin"):
        if not os.path.exists("./trained_model/tokenizer.json"):
            train_tokenizer(
                file_name_no_bos,
                vocab_size=vocab_size,
                serialize=False,
                save_path="trained_model",
            )
        reload = False

    elif os.path.exists("./trained_model/pytorch_model.bin"):
        print("reloading model")
        reload = True

    config = GPT2ConfigCPU(
        vocab_size=vocab_size,
        max_length=max_length,
        n_embed=n_embed,
        n_layer=n_layer,
        n_head=n_head,
        bos_token_id=0,
        eos_token_id=0,
        dropout=dropout,
        learning_rate=learning_rate,
    )
    t_list = token_chunk_split(
        file_name_with_bos,
        trim_size=max_length,
        split_string=split_string
    )
    data = TokenDataset(
        texts=t_list,
        tokenizer_file="./trained_model/tokenizer.json",
        block_size=max_length,
        line_by_line=line_by_line,
        save_cache=False,
    )

    if reload is True:
        ai = aitextgen(
            model_folder="./trained_model",
            config="./trained_model/config.json",
            tokenizer_file="./trained_model/tokenizer.json",
            vocab_file="./trained_model/vocab.json",
        )
    else:
        ai = aitextgen(
            config=config,
            tokenizer_file="./trained_model/tokenizer.json",
            vocab_file="./trained_model/vocab.json",
        )

    print()
    print(f"TokenDataset containing {len(t_list)} subsets loaded from file.")
    num_steps = floor(epochs * (len(t_list) / batch_size))
    print(f"Epochs set to {epochs}, so num_steps set to {num_steps}.")
    print(f"1 epoch would be {floor((len(t_list) / batch_size))} steps.")
    print()

    # TODO: make this remove beginning of training data to resume properly.
    # if reload is True:
    #     with open('trained_model/step.txt', 'r') as file:
    #         stepped = int(file.read())
    #     if stepped > 0:
    #         print(
    #             f"Resuming Training. First {stepped} of "
    #             f"dataset skipped. First {num_steps} skipped."
    #         )
    #         print(type(data))
    #         data = data[stepped:]
    #         num_steps = num_steps - stepped



    # Initialize Weights & Biases
    wandb.init(
        project=wandb_project_name,
        config={
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "max_length": max_length,
            "n_embed": n_embed,
            "n_layer": n_layer,
            "n_head": n_head,
            "dropout": dropout,
            "vocab_size": vocab_size,
        }
    )

    # Train the model! It will save pytorch_model.bin periodically and after completion to the
    # `trained_model` folder.
    ai.train(
        data,
        batch_size=batch_size,
        num_steps=num_steps,
        save_every=save_every,
        generate_every=generate_every,
        learning_rate=learning_rate,
    )

    wandb.finish()

    # Generate text from it!
    ai.generate(10, prompt="<|thread|>")


if __name__ == "__main__":
    main()
