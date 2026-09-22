from transformers import AutoTokenizer

from tokenizer.tokenizer import QwenTokenizer


def test_tokenizer_data():


    tokenizer = QwenTokenizer(
    "models/Qwen2.5-0.5B/vocab.json",
    "models/Qwen2.5-0.5B/merges.txt"
    )

    print('\n')
    print(len(tokenizer.token_to_id))
    print(len(tokenizer.merge_rank))
    print(tokenizer.token_to_id['!'])
    print(tokenizer.id_to_token[0])

def test_tokenizer_decode():
    MODEL_PATH = "models/Qwen2.5-0.5B"

    hf = AutoTokenizer.from_pretrained(MODEL_PATH)

    my = QwenTokenizer(f"{MODEL_PATH}/vocab.json", f"{MODEL_PATH}/merges.txt")

    text = "Hello world!"

    ids = hf.encode(text)

    hf_txt = hf.decode(ids)
    my_txt = my.decode(ids)

    assert my_txt == hf_txt

def test_tokenizer_hf():
    MODEL_PATH = "models/Qwen2.5-0.5B"

    hf = AutoTokenizer.from_pretrained(MODEL_PATH)

    my = QwenTokenizer(f"{MODEL_PATH}/vocab.json", f"{MODEL_PATH}/merges.txt")

    tests = ["Hello world!",
             "The capital of France is Paris.",
             "hello",
             " hello",
             "hello ",
             "hello     world",
             "I'm testing.  !",
             "we're good",
             "123456789",
             "!!!",
             "hello!!!",
             "\nHello\nworld\n",
             "café",
             "é",
             "e\u0301",
             "тест123",
             "轼津儱祄芧",
             "🙂",
             "🙂 🙂\'🙂",
             "a+b*c-d=4",
             "",]

    for text in tests:
        hf_ids = hf.encode(text)
        my_ids = my.encode(text)

        print(repr(text))
        print("hf:", hf_ids)
        print("test:", my_ids)

        assert my_ids == hf_ids