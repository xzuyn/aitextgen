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
        # multiple_sets: bool = False,  # TODO
        # multiple_sets_amount: int = 2,  # TODO
        # randomize_sets: bool = False,  # TODO
        # randomize_before_merge: bool = False,  # TODO
        sanity_check: bool = False,
):
    """
    :param file_path: your text file.
    :param trim: tokenize your text chunks then only keeps chunks that are trim_size or smaller.
    :param trim_size: should set this to your models `max_length`.
    :param split_string: what is seperating your text chunks currently? if its one thing on each line then using `\\n` will work. if you have something else like `<|thread|>` or `<|bos|>`, then use those instead.
    :param prefix: add a prefix to your message chunks (will be added before tokenizing).
    :param suffix: add a suffix to your message chunks (will be added before tokenizing).
    :param keep_split_string: when set to False your split string will be removed. so its probably best to keep this set to True.
    :param breaks_before_chunks: in between your `split_string` and text chunk you can add line breaks, or really any text if you would like.
    :param tokenizer_file: your models tokenizer file.
    :param config_file: your models config file.
    :param fasttokenizer: swap between using `PreTrainedTokenizerFast` & `PreTrainedTokenizer`
    :param resume_step: set how many steps worth of data to remove from the beginning of your dataset (this happens in memory. your original dataset file will stay the same).
    :param sanity_check: will load the first item of your message chunks, and print it out. this is so you can double check if you are sending the trainer the correct formatting. triple check this.

    The final output message chunks will be from these;
    {prefix}{split_string}{breaks_before_chunk}{chunks}{suffix}
    """
    global rerechunked, tokenizer
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        chunks = content.split(split_string)
    content = None

    message_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]
    chunks = None

    rechunked = []
    if keep_split_string is True:
        for chunks in message_chunks:
            rechunked.append(f"{prefix}{split_string}{breaks_before_chunk}{chunks}{suffix}")
    elif keep_split_string is False:
        for chunks in message_chunks:
            rechunked.append(f"{prefix}{breaks_before_chunk}{chunks}{suffix}")
    message_chunks = None

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
        if sanity_check is True:  # TODO: add more to this
            for i in rechunked:
                tokens = tokenizer.encode(i)
                if len(tokens) <= trim_size:
                    rerechunked.append(i)
                    ## Sanity check
                    print("--- YOUR INPUT DIRECTLY UNDER THIS ---")
                    print(i)
                    print("--- YOUR INPUT DIRECTLY ABOVE THIS ---")
                    exit()

        elif sanity_check is False:
            for i in rechunked:
                tokens = tokenizer.encode(i)
                if len(tokens) <= trim_size:
                    rerechunked.append(i)
        tokens = None
        rechunked = None

    return rerechunked


def plaintext_slider(
        file_path,
        size,
        tokenizer_file: str = "./trained_model/tokenizer.json",
        config_file: str = "./trained_model/config.json",
        fasttokenizer: bool = True,
):
    if fasttokenizer is True:
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast.from_pretrained(tokenizer_file,
                                                            config=config_file)
    elif fasttokenizer is False:
        from transformers import PreTrainedTokenizer
        tokenizer = PreTrainedTokenizer.from_pretrained(tokenizer_file,
                                                        config=config_file)

    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()
        tokens_list = tokenizer.encode(content)
        content = None

    grouped_tokens = []
    for i in range(0, len(tokens_list) - (size - 1)):
        group = tokens_list[i:i + size]
        grouped_tokens.append(group)
    group = None
    tokens_list = None

    plaintext_list = [tokenizer.decode(tokens) for tokens in grouped_tokens]
    grouped_tokens = None
    return plaintext_list
