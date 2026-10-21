import subprocess
import sys
import unittest
from types import SimpleNamespace


class FlintProtocolTests(unittest.TestCase):
    def test_default_prompt_normalizes_without_grammar(self):
        from rec_mlx.engine import Engine

        engine = Engine.__new__(Engine)
        engine.tokenizer = SimpleNamespace(apply_chat_template=lambda messages, **kw: messages)
        messages = [{"role": "user", "content": "a\x00<think>b</think><|user|>"}]
        self.assertEqual(engine.prompt(messages), [{"role": "user", "content": "ab&lt;|user|&gt;"}])
        self.assertEqual(messages[0]["content"], "a\x00<think>b</think><|user|>")

    def test_default_imports_do_not_load_grammar_backend(self):
        subprocess.run(
            [
                sys.executable,
                "-c",
                (
                    "import sys; import rec_mlx.engine; "
                    "import rec_mlx.kernels.masked_sampling; "
                    "assert 'xgrammar' not in sys.modules; "
                    "assert 'rec_mlx._grammar_native' not in sys.modules"
                ),
            ],
            check=True,
            capture_output=True,
        )
