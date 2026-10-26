"""The password in front of the public dashboard.

What is being checked is the property the deployment depends on: with the
variable set, nothing after require_password() runs until the right password
arrives. Streamlit's own test runner is used so that `st.stop()` means what it
means in the app rather than what a fake would make it mean.
"""

import pytest
from streamlit.testing.v1 import AppTest

from policygate.viz.auth import ENV_VAR

SECRET = "the-password"

SCRIPT = """
import streamlit as st
from policygate.viz.auth import require_password

require_password()
st.write("the dashboard")
"""


def app(monkeypatch, password=SECRET):
    if password is None:
        monkeypatch.delenv(ENV_VAR, raising=False)
    else:
        monkeypatch.setenv(ENV_VAR, password)
    return AppTest.from_string(SCRIPT).run()


def body_shown(at):
    return any("the dashboard" in str(m.value) for m in at.markdown)


def test_without_the_variable_there_is_no_gate(monkeypatch):
    """Local runs read local files and talk to localhost; a password there
    protects nothing from anyone."""
    at = app(monkeypatch, password=None)
    assert body_shown(at)
    assert not at.text_input


def test_the_page_is_not_reachable_until_the_password_is_given(monkeypatch):
    at = app(monkeypatch)
    assert not body_shown(at)
    assert at.text_input[0].label == "Password"


def test_the_wrong_password_says_so_and_keeps_the_page_shut(monkeypatch):
    at = app(monkeypatch)
    at.text_input[0].input("not it")
    at.button[0].click().run()

    assert not body_shown(at)
    assert "not the password" in at.error[0].value


def test_the_right_password_opens_it_and_it_stays_open(monkeypatch):
    at = app(monkeypatch)
    at.text_input[0].input(SECRET)
    at.button[0].click().run()

    assert body_shown(at)
    # The next rerun must not ask again: session state carries the unlock.
    at.run()
    assert body_shown(at)


@pytest.mark.parametrize("wrong", ["", "the-passwor", "the-password ", "THE-PASSWORD"])
def test_near_misses_are_misses(monkeypatch, wrong):
    at = app(monkeypatch)
    at.text_input[0].input(wrong)
    at.button[0].click().run()
    assert not body_shown(at)
