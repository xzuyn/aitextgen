from aitextgen import aitextgen

ai = aitextgen(model_folder="./trained_model",
               tokenizer_file="./trained_model/tokenizer.json",
               config="./trained_model/config.json",
               vocab_file="./trained_model/vocab.json",
               merges_file="./trained_model/merges.txt"
               )

while True:
    sentence = str(input("Message: "))
    ai.generate(
        1,
        prompt=sentence,
        temperature=0.7,
        max_length=1024
    )
