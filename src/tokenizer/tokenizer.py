import json
import unicodedata
import regex as re
import torch

class QwenTokenizer:

    PRETOKENIZE_REGEX = (r"(?i:'s|'t|'re|'ve|'m|'ll|'d)"
                         r"|[^\r\n\p{L}\p{N}]?\p{L}+"
                         r"|\p{N}"
                         r"| ?[^\s\p{L}\p{N}]+[\r\n]*"
                         r"|\s*[\r\n]+"
                         r"|\s+(?!\S)"
                         r"|\s+")

    def __init__(self, vocab_path, merges_path):

        with open(vocab_path, "r", encoding="utf-8") as f:
            self.token_to_id = json.load(f)

        self.id_to_token = {
            token_id: token
            for token, token_id in self.token_to_id.items()
        }

        with open(merges_path, "r", encoding="utf-8") as f:
            lines = f.readlines()

        self.merge_rank = {}

        rank = 0
        for line in lines:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            left, right = line.split()

            self.merge_rank[(left, right)] = rank
            rank += 1

        self.pattern = re.compile(self.PRETOKENIZE_REGEX)
        self.byte_char_dict = self.bytes_to_unicode()
        self.char_byte_dict = {char: byte for byte, char in self.byte_char_dict.items()}
        self.bpe_cache = {}


    def bytes_to_unicode(self):
        bytes = (list(range(ord("!"), ord("~") + 1))
            + list(range(ord("¡"), ord("¬") + 1))
            + list(range(ord("®"), ord("ÿ") + 1)))

        chars = bytes.copy()

        n = 0

        for b in range(256):
            if b not in bytes:
                bytes.append(b)
                chars.append(256 + n)
                n += 1

        chars = [chr(c) for c in chars]

        return dict(zip(bytes, chars))

    def pretokenize(self, text):
        return self.pattern.findall(text)

    def _bpe(self, text):

        symbols = list(text)

        while len(symbols) > 1:

            mpair = None
            mrank = float("inf")

            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])

                rank = self.merge_rank.get(pair)

                if rank is not None and rank < mrank:
                    mrank = rank
                    mpair = pair

            if mpair is None:
                break

            new_sym = []
            i = 0

            while i < len(symbols):

                if (i + 1 < len(symbols) and symbols[i] == mpair[0] and symbols[i + 1] == mpair[1]):
                    new_sym.append(symbols[i] + symbols[i + 1])
                    i += 2
                else:
                    new_sym.append(symbols[i])
                    i += 1

            symbols = new_sym

        return symbols

    def bpe(self, text):
        if text in self.bpe_cache:
            return self.bpe_cache[text]

        result = self._bpe(text)

        self.bpe_cache[text] = result
        return result

    def encode(self, text):
        text = unicodedata.normalize("NFC", text)

        matches = self.pattern.findall(text)
        token_ids = []

        for match in matches:

            chars = "".join(self.byte_char_dict[b] for b in match.encode("utf-8"))

            bpe_tokens = self.bpe(chars)

            for token in bpe_tokens:
                token_ids.append(self.token_to_id[token])

        return token_ids

    def decode(self, token_ids):

        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.flatten().tolist()

        text = "".join(self.id_to_token[token_id] for token_id in token_ids)

        byte_values = [self.char_byte_dict[c]  for c in text]

        return bytes(byte_values).decode("utf-8", errors="replace")
            