from tokenizers.decoders import DecodeStream


class StreamingText:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stream = DecodeStream(skip_special_tokens=False)
        self.emitted = 0

    def add_token(self, token):
        text = self.stream.step(self.tokenizer.backend_tokenizer, token) or ""
        self.emitted += len(text)
        return text

    def finish(self, tokens):
        # Decode the whole answer only here, to flush incomplete UTF-8; per-token re-decoding is quadratic.
        text = self.tokenizer.decode(tokens, skip_special_tokens=False)
        tail = text[self.emitted :]
        self.emitted = len(text)
        return tail
