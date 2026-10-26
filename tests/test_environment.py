"""Tests for the project's .env handling.

The behaviour under test is the one that bit a live session: a shell exporting
another project's OPENAI_API_KEY shadowed this project's, and the first model
call came back 401 from a provider that key was never meant to reach.
"""

import pytest

from policygate.environment import apply_project_env, describe, parse_env_file


class TestParsing:
    def test_the_shapes_a_dotenv_actually_uses(self):
        parsed = parse_env_file(
            "\n".join(
                [
                    "# a comment",
                    "",
                    "OPENAI_API_KEY=sk-abc",
                    "export OPENAI_API_BASE=https://gateway.example/v1",
                    'QUOTED="with spaces"',
                    "SINGLE='single'",
                    "  SPACED = padded ",
                    "not a definition",
                ]
            )
        )
        assert parsed == {
            "OPENAI_API_KEY": "sk-abc",
            "OPENAI_API_BASE": "https://gateway.example/v1",
            "QUOTED": "with spaces",
            "SINGLE": "single",
            "SPACED": "padded",
        }

    def test_a_value_containing_an_equals_sign_survives(self):
        assert parse_env_file("URL=a=b=c") == {"URL": "a=b=c"}


class TestApplying:
    def test_the_file_wins_over_an_exported_variable(self, tmp_path, monkeypatch):
        """The whole point: load_dotenv would leave the shell's value in place."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-from-the-shell")
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-the-projects-own\n")

        changed = apply_project_env(env)

        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-the-projects-own"
        assert changed == ["OPENAI_API_KEY"]

    def test_it_reports_only_what_it_actually_changed(self, tmp_path, monkeypatch):
        monkeypatch.setenv("ALREADY", "same")
        env = tmp_path / ".env"
        env.write_text("ALREADY=same\nNEW=value\n")
        assert apply_project_env(env) == ["NEW"]

    def test_no_file_is_not_an_error(self, tmp_path):
        """In a container the environment is the configuration and there is no file."""
        assert apply_project_env(tmp_path / "absent") == []

    def test_it_can_be_refused(self, tmp_path, monkeypatch):
        monkeypatch.setenv("POLICYGATE_IGNORE_DOTENV", "1")
        monkeypatch.setenv("OPENAI_API_KEY", "sk-deliberate")
        env = tmp_path / ".env"
        env.write_text("OPENAI_API_KEY=sk-from-file\n")

        assert apply_project_env(env) == []

        import os

        assert os.environ["OPENAI_API_KEY"] == "sk-deliberate"


@pytest.mark.parametrize(
    "changed,expected",
    [([], "no variables taken from .env"), (["B", "A"], "overriding the environment): A, B")],
)
def test_it_says_what_it_did(changed, expected):
    assert expected in describe(changed)
