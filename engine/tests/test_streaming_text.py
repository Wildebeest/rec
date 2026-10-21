import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, decoders, models, pre_tokenizers
from transformers import PreTrainedTokenizerFast

from rec_mlx.detokenize import StreamingText


def tokenizer():
    vocab = {
        value: index for index, value in enumerate(sorted(pre_tokenizers.ByteLevel.alphabet()))
    }
    backend = Tokenizer(models.BPE(vocab, []))
    backend.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    backend.decoder = decoders.ByteLevel()
    backend.add_special_tokens(["<think>", "</think>", "<tool_call>", "<eos>"])
    return PreTrainedTokenizerFast(tokenizer_object=backend, clean_up_tokenization_spaces=False)


class StreamingTextTests(unittest.TestCase):
    def test_matches_full_decode_at_every_termination_boundary(self):
        tok = tokenizer()
        ids = tok.encode('  <think>café 🦊</think>\n<tool_call>{"path":"雪.py"}\n')
        for stop in range(1, len(ids) + 1):
            stream = StreamingText(tok)
            pieces = [stream.add_token(token) for token in ids[:stop]]
            pieces.append(stream.finish(ids[:stop]))
            self.assertEqual("".join(pieces), tok.decode(ids[:stop], skip_special_tokens=False))

    def test_no_whole_output_decode_until_final_flush(self):
        tok = tokenizer()
        ids = tok.encode("a" * 8192)
        stream = StreamingText(tok)
        with patch.object(tok, "decode", wraps=tok.decode) as decode:
            pieces = [stream.add_token(token) for token in ids]
            decode.assert_not_called()
            pieces.append(stream.finish(ids))
            self.assertEqual(decode.call_count, 1)
        self.assertEqual("".join(pieces), "a" * 8192)


if __name__ == "__main__":
    unittest.main()
