# SPDX-License-Identifier: Apache-2.0
"""The chat-template file fallback: checkpoints that ship the template as a
standalone file (chat_template.jinja / chat_template.json) must yield a
usable template even when AutoTokenizer does not pick it up."""

from agrimlxo.tokenizer import _read_template_file


def test_jinja_file(tmp_path):
    (tmp_path / "chat_template.jinja").write_text(
        "{% for m in messages %}{{ m['content'] }}{% endfor %}"
    )
    assert "messages" in _read_template_file(tmp_path)


def test_json_single_template(tmp_path):
    (tmp_path / "chat_template.json").write_text(
        '{"chat_template": "{% for m in messages %}X{% endfor %}"}'
    )
    assert _read_template_file(tmp_path).endswith("endfor %}")


def test_json_multi_template_prefers_default(tmp_path):
    (tmp_path / "chat_template.json").write_text(
        '[{"name": "reasoning", "template": "R"},'
        ' {"name": "default", "template": "D"}]'
    )
    assert _read_template_file(tmp_path) == "D"


def test_json_multi_template_no_default_falls_back_to_first(tmp_path):
    (tmp_path / "chat_template.json").write_text(
        '[{"name": "a", "template": "A"}, {"name": "b", "template": "B"}]'
    )
    assert _read_template_file(tmp_path) == "A"


def test_missing_files_return_none(tmp_path):
    assert _read_template_file(tmp_path) is None
