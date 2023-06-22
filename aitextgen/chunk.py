def token_chunk_split(
        file_path,
        trim: bool = True,
        trim_size: int = 1024,
        split_string: str = "\n",
        keep_split_string: bool = True,
        add_extra_linebreak: bool = True,
        tokenizer_file: str = "./trained_model/tokenizer.json",
        config_file: str = "./trained_model/config.json",
        fasttokenizer: bool = True,
        line_by_line_cust: bool = False,
        prefix: str = None,
):
    global rerechunked, tokenizer
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    chunks = content.split(split_string)
    message_chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    rechunked = []
    if add_extra_linebreak is True:
        for chunks in message_chunks:
            rechunked.append(f"<|thread|>\n{chunks}")
    elif add_extra_linebreak is False:
        for chunks in message_chunks:
            rechunked.append(f"<|thread|>\n\n{chunks}")

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

    return rerechunked
