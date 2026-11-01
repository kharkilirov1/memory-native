"""Tests for FakeAD (AgentDesktop is tested indirectly through the env tests)."""

from deskenv.ad import FakeAD, ADError, _ScriptedStep


def test_fakead_basic_snapshot():
    ad = FakeAD([_ScriptedStep(snapshot={"ok": True, "data": {"tree": {"role": "window"}}})])
    result = ad.snapshot("mousepad")
    assert result["ok"] is True
    assert result["data"]["tree"]["role"] == "window"


def test_fakead_act():
    ad = FakeAD([_ScriptedStep(act={"ok": True})])
    result = ad.act("click", "@s1:e1", app="mousepad")
    assert result["ok"] is True


def test_fakead_ok_false_passthrough():
    """FakeAD returns ok:false dict; it does not raise."""
    ad = FakeAD([_ScriptedStep(act={"ok": False, "error": {"code": "STALE_REF"}})])
    result = ad.act("click", "@s1:e1")
    assert result["ok"] is False
    assert result["error"]["code"] == "STALE_REF"


def test_fakead_raises_on_out_of_steps():
    ad = FakeAD([])
    try:
        ad.snapshot("mousepad")
        assert False, "expected ADError"
    except ADError as e:
        assert "no more scripted steps" in str(e)


def test_fakead_raises_scripted_error():
    ad = FakeAD([_ScriptedStep(error="connection refused")])
    try:
        ad.snapshot("mousepad")
        assert False, "expected ADError"
    except ADError as e:
        assert "connection refused" in str(e)


def test_fakead_call_log():
    ad = FakeAD([
        _ScriptedStep(snapshot={"ok": True}),
        _ScriptedStep(act={"ok": True}),
    ])
    ad.snapshot("mousepad")
    ad.act("click", "@s1:e1", app="mousepad")
    assert len(ad.call_log) == 2
    assert ad.call_log[0] == ("snapshot", ("mousepad",), {})
    assert ad.call_log[1] == ("act", ("click", "@s1:e1"), {"app": "mousepad"})


def test_fakead_reset():
    ad = FakeAD([_ScriptedStep(snapshot={"ok": True})])
    ad.snapshot("mousepad")
    ad.reset([_ScriptedStep(snapshot={"ok": False})])
    result = ad.snapshot("mousepad")
    assert result["ok"] is False
    assert ad._pos == 1


def test_fakead_wrong_caller_step():
    """If a step has only snapshot data but act is called, raise."""
    ad = FakeAD([_ScriptedStep(snapshot={"ok": True})])
    try:
        ad.act("click", "@x")
        assert False, "expected ADError"
    except ADError as e:
        assert "no response for act" in str(e)
