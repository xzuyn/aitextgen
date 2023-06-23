def token_chunk_split(
        file_path,
        trim: bool = True,
        trim_size: int = 1024,
        split_string: str = "\n",
        prefix: str = "",
        suffix: str = "",
        keep_split_string: bool = True,
        breaks_before_chunk: str = "\n",
        tokenizer_file: str = "./trained_model/tokenizer.json",
        config_file: str = "./trained_model/config.json",
        fasttokenizer: bool = True,
        resume_step: int = 0,
):
    """
    file_path: your text file.

    trim: tokenize your text chunks then only keeps chunks that are trim_size or smaller.

    trim_size: should set this to your models `max_length`.

    split_string: what is seperating your text chunks currently? if its one thing on each line
    then using `\\n` will work. if you have something else like `<|thread|>` or `<|bos|>`, then use
    those instead.

    prefix: add a prefix to your message chunks (will be added before tokenizing).

    suffix: add a suffix to your message chunks (will be added before tokenizing).

    keep_split_string: when set to False your split string will be removed.

    so its probably best to keep this set to True.

    breaks_before_chunks: in between your `split_string` and text chunk you can add line breaks,
    or really any text if you would like.

    tokenizer_file: your models tokenizer file.

    config_file: your models config file.

    fasttokenizer: swap between using `PreTrainedTokenizerFast` & `PreTrainedTokenizer`

    resume_step: set how many steps worth of data to remove from the beginning of your dataset
    (this happens in memory. your original dataset file will stay the same).

    The final output message chunks will be from these;
    {prefix}{split_string}{breaks_before_chunk}{chunks}{suffix}
    """
    global rerechunked, tokenizer
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    chunks = content.split(split_string)
    message_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    rechunked = []
    for chunks in message_chunks:
        rechunked.append(f"{prefix}{split_string}{breaks_before_chunk}{chunks}{suffix}")

    rechunked = rechunked[resume_step:]

    if trim is True:
        rerechunked = []
        if fasttokenizer is True:
            from transformers import PreTrainedTokenizerFast
            tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file,
                                                                config=config_file)
        elif fasttokenizer is False:
            from transformers import PreTrainedTokenizer
            tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_file,
                                                                config=config_file)
        for i in rechunked:
            tokens = tokenizer.encode(i)
            if len(tokens) <= trim_size:
                rerechunked.append(i)

                ## Sanity check
                print(i)
                exit()

    return rerechunked
