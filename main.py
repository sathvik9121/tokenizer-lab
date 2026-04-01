from helper import get_openai_api_key

api_key = get_openai_api_key()
print(api_key)


import warnings
warnings.filterwarnings('ignore')

from helper import get_openai_api_key

api_key = get_openai_api_key()
print("API Key loaded:", api_key)

from transformers import AutoTokenizer

print("Transformers library is working ✅")

import warnings
warnings.filterwarnings('ignore')

from transformers import AutoTokenizer

sentence = "This type of tokenizer splits a text into individual characters This is often used for tasks such as text classification or sentiment analysis"

tokenizer = AutoTokenizer.from_pretrained("bert-base-cased")

tokens = tokenizer.tokenize(sentence)

token_ids = tokenizer.convert_tokens_to_ids(tokens)

encoded = tokenizer(sentence)

print("Original Sentence:", sentence)
print("Tokens:", tokens)
print("Token IDs:", token_ids)
print("Encoded Output:", encoded)

colors = [
    '102;194;165', '252;141;98', '141;160;203',
    '231;138;195', '166;216;84', '255;217;47'
]

def show_tokens(sentence: str, tokenizer):
    token_ids = tokenizer(sentence).input_ids

    print(f"\nVocab length: {len(tokenizer)}\n")

    for idx, t in enumerate(token_ids):
        print(
            f'\x1b[0;30;48;2;{colors[idx % len(colors)]}m' +
            tokenizer.decode(t) +
            '\x1b[0m',
            end=' '
        )
    print("\n")

show_tokens(sentence, tokenizer)


