"""Tests for the standalone public build: data directory, process detection, alerts."""
import importlib
import subprocess
import sys

import pytest

from augur import core as A


def test_empty_augur_home_falls_back_to_the_default_not_the_cwd(monkeypatch):
    monkeypatch.setenv("AUGUR_HOME", "")
    mod = importlib.reload(A)
    try:
        assert mod.DATA_DIR.name == ".augur" and mod.DATA_DIR.is_absolute()
    finally:
        monkeypatch.delenv("AUGUR_HOME", raising=False)
        importlib.reload(A)


def test_augur_home_is_honoured(monkeypatch, tmp_path):
    monkeypatch.setenv("AUGUR_HOME", str(tmp_path / "h"))
    mod = importlib.reload(A)
    try:
        assert mod.DATA_DIR == tmp_path / "h"
    finally:
        monkeypatch.delenv("AUGUR_HOME", raising=False)
        importlib.reload(A)


def _sleeper(*argv):
    return subprocess.Popen([sys.executable, "-c", "import time; time.sleep(5)", *argv])


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="reads /proc cmdline")
def test_process_check_matches_an_augur_day_run_only():
    real, lookalike, other = _sleeper("augur", "day"), _sleeper("augurize", "day"), _sleeper("augur", "bench")
    try:
        assert A._pid_alive(real.pid)
        assert not A._pid_alive(lookalike.pid)       # a substring is not an Augur run
        assert not A._pid_alive(other.pid)           # an Augur process that is not a day run
    finally:
        for p in (real, lookalike, other):
            p.kill()


def test_alert_without_a_channel_is_recorded_not_raised(monkeypatch):
    for k in ("AUGUR_SMTP_HOST", "AUGUR_SMTP_USER", "AUGUR_SMTP_PASSWORD", "AUGUR_ALERT_TO"):
        monkeypatch.delenv(k, raising=False)
    ok, why = A._email("s", "b")
    assert ok is False and "no alert channel" in why


def test_alert_uses_starttls_when_asked(monkeypatch):
    calls = []

    class _SMTP:
        def __init__(self, host, port, timeout, context=None):
            calls.append(("ssl" if context is not None else "plain", host, port))
            self.ctx = context

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def starttls(self, context=None):
            calls.append(("starttls", context is not None and context.check_hostname
                          and context.verify_mode == ssl.CERT_REQUIRED))

        def login(self, u, p):
            calls.append(("login", u))

        def send_message(self, m):
            calls.append(("sent", m["To"]))

    import smtplib
    import ssl
    monkeypatch.setattr(smtplib, "SMTP", _SMTP)
    monkeypatch.setattr(smtplib, "SMTP_SSL", _SMTP)
    for k, v in {"AUGUR_SMTP_HOST": "smtp.example.org", "AUGUR_SMTP_USER": "me@example.org",
                 "AUGUR_SMTP_PASSWORD": "x", "AUGUR_ALERT_TO": "you@example.org",
                 "AUGUR_SMTP_STARTTLS": "1"}.items():
        monkeypatch.setenv(k, v)
    ok, why = A._email("s", "b")
    # STARTTLS with a context that checks the certificate and the host name
    assert ok and calls[0] == ("plain", "smtp.example.org", 587) and ("starttls", True) in calls
    calls.clear()
    monkeypatch.setenv("AUGUR_SMTP_STARTTLS", "0")
    ok, why = A._email("s", "b")
    assert ok and calls[0] == ("ssl", "smtp.example.org", 465)
