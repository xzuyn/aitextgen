import os
import wandb
import logging
from math import floor
from aitextgen import aitextgen
from aitextgen.utils import GPT2ConfigCPU
from aitextgen.chunk import token_chunk_split
from aitextgen.TokenDataset import TokenDataset
from aitextgen.tokenizers import train_tokenizer


batch_size = 1  # high batch size gives worse training(?), but is used to speed up training time
max_length = 64  # default is 64
epochs = 4  # default is 1
save_every = 5000  # default is 5000
generate_every = 500  # default is 500
vocab_size = (1024 * 1)  # default is (1024 * 1)
n_embed = 128  # default is 128
n_layer = 4  # default is 4
n_head = 4  # default is 4
line_by_line = False  # default is False
learning_rate = 0.001  # default is 0.001
dropout = 0.0  # default is 0.0
split_string = "<|endoftext|>"
keep_split_string = False
trim = True
fasttokenizer = True
breaks_before_chunk = ""
suffix = "<|endoftext|>"
prefix = ""
tokenizer_file = "./trained_model/tokenizer.json"
config_file = "./trained_model/config.json"
vocab_file = "./trained_model/vocab.json"
prompt = "USER: "
wandb_update_rate = 5  # number of steps until sending loss info to wandb. currently does nothing

# this will load the first chunk of your dataset so can can see if it's using the correct format.
# your model will not train with this enabled. it will not get saved either.
# just enabled this to see what the trainer will see.
sanity_check = False


stepped = 0

# Trying to train a tokenizer with your <|endoftext|> token in the dataset
# will cause it to break it into multiple tokens, while also being a special token.
# So we train the tokenizer on a dataset without it to make sure it stays as a single special token.
# We will train the actual model on the no_bos version.
#
# file_name_with_bos = Your dataset WITH <|endoftext|> tokens.
# file_name_no_bos = Your dataset WITHOUT <|endoftext|> tokens.
file_name_with_bos = "./Datasets/SlimPajamaTestSet.txt"
file_name_no_bos = "./Datasets/SlimPajamaTestSet-no-bos.txt"  # TODO: make this automatic
wandb_project_name = "SlimPajamaTestSet"
wandb_run_name = "SlimPajamaTestSet"  # Dial-EPOCH-1


# Your code needs to be wrapped inside a main function,
# as otherwise multiple child processes from pytorch_lightning cannot be spawned
def main():
    # Setup logging
    global reload, stepped
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

    if reload is True and os.path.exists("./trained_model/step.txt"):
        with open('./trained_model/step.txt', 'r', encoding="utf-8") as file:
            stepped = int(file.read())
        if stepped > 0:
            print(
                f"Resuming Training. First {stepped} sets of the "
                f"dataset skipped. First {stepped} steps skipped."
            )

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
        trim=trim,
        trim_size=max_length,
        split_string=split_string,
        tokenizer_file=tokenizer_file,
        config_file=config_file,
        breaks_before_chunk=breaks_before_chunk,
        fasttokenizer=fasttokenizer,
        resume_step=stepped,
        sanity_check=sanity_check,
        suffix=suffix,
        prefix=prefix,
        keep_split_string=keep_split_string,
    )
    data = TokenDataset(
        texts=t_list,
        tokenizer_file=tokenizer_file,
        block_size=max_length,
        line_by_line=line_by_line,
        save_cache=False,  # IDK something weird was happening, so I don't save it
    )

    if reload is True:
        ai = aitextgen(
            model_folder="./trained_model",
            config=config_file,
            tokenizer_file=tokenizer_file,
            vocab_file=vocab_file,
        )
    else:
        ai = aitextgen(
            config=config,
            tokenizer_file=tokenizer_file,
            vocab_file=vocab_file,
        )

    print()
    print(f"TokenDataset containing {len(t_list)} subsets loaded from file.")
    num_steps = floor(epochs * (len(t_list) / batch_size))
    print(f"Epochs set to {epochs}, so num_steps set to {num_steps}.")
    print(f"1 epoch would be {floor((len(t_list) / batch_size))} steps.")
    print()

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
    ai.generate(
        10,
        prompt=prompt,
    )


if __name__ == "__main__":
    main()
