# aitextgen

A robust tool for advanced AI text generation.

aitextgen is a Python package that leverages Huggingface Transformers and PyTorch with specific optimizations and added features. It is the successor to textgenrnn and gpt-2-simple, merging the advantages of both packages.

* Finetunes on a pretrained GPT-2 model...or create your own architecture and train from scratch!
* Generates text faster than gpt-2-simple and with better memory efficiency
* Preserves compatibility with Transformers, allowing you to use the model for other NLP tasks.
* Model agnostic and future-proofed to support new developments in the Transformers-based world.

## Installation

## Usage

## Helpful Notes

## To-Do

* Add native TensorFlow 2.0 support (dependent on the TF 2.0 generation/finetuning support in Transformers)

## Maintainer/Creator

Max Woolf ([@minimaxir](https://minimaxir.com))

*Max's open-source projects are supported by his [Patreon](https://www.patreon.com/minimaxir) and [GitHub Sponsors](https://github.com/sponsors/minimaxir). If you found this project helpful, any monetary contributions to the Patreon are appreciated and will be put to good creative use.*

## License

MIT

Code from Transformers is used following its Apache License 2.0, with state changes noted when relevant.