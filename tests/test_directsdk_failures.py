"""Native failures must be diagnosable and must not discard a usable reply.

- A native child that dies says why only on stderr; that text must reach the error.
- A valid reply naming a tool this turn does not offer is handed to Hermes (which answers the
  model with a tool error) instead of being thrown away and re-asked from scratch.
- ``create()`` on a thread with a running loop must still serve a synchronous caller: Relay
  drives sync provider callbacks from inside a task, and a bare coroutine there reached title
  generation as "invalid response" and moved it to the fallback provider.
"""
import asyncio
import json
import os
from pathlib import Path
import sys
import tempfile

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

DIES_ON_REPLAY = r"""
import sys
for line in sys.stdin:
    sys.stderr.write('Error: replay frame rejected: bad content block\n'); sys.stderr.flush()
    sys.exit(3)
"""

UNKNOWN_TOOL = r"""
import json, sys
for line in sys.stdin:
    pass
blocks = [{'type': 'text', 'text': 'calling'}, {'type': 'tool_use', 'id': 'toolu_x', 'name': 'mcp__hermes__process_manage', 'input': {'action': 'poll'}}]
print(json.dumps({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'calling'}}}), flush=True)
print(json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': blocks, 'id': 'msg_x', 'model': 'sonnet', 'stop_reason': 'tool_use'}}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'num_turns': 2, 'subtype': 'error_max_turns', 'is_error': True, 'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
sys.exit(1)
"""

PLAIN = r"""
import json, sys
for line in sys.stdin:
    pass
print(json.dumps({'type': 'stream_event', 'event': {'type': 'content_block_delta', 'delta': {'type': 'text_delta', 'text': 'hi'}}}), flush=True)
print(json.dumps({'type': 'assistant', 'message': {'role': 'assistant', 'content': [{'type': 'text', 'text': 'hi'}], 'id': 'msg_p', 'model': 'sonnet', 'stop_reason': 'end_turn'}}), flush=True)
print(json.dumps({'type': 'stream_event', 'event': {'type': 'message_stop'}}), flush=True)
print(json.dumps({'type': 'result', 'num_turns': 1, 'subtype': 'success', 'is_error': False, 'usage': {'input_tokens': 1, 'output_tokens': 1}}), flush=True)
"""


def _client(tmp, script):
    import directsdk

    path = Path(tmp) / "native.py"
    path.write_text(script)
    return directsdk.Client(command=[sys.executable, str(path)], env={"PATH": os.environ["PATH"], "HOME": tmp}, timeout=30)


def _tool(name):
    return {"type": "function", "function": {"name": name, "description": "d", "parameters": {"type": "object", "properties": {}}}}


def test_native_exit_during_replay_carries_exit_code_and_stderr():
    history = [
        {"role": "user", "content": "first"},
        {"role": "assistant", "content": "ok"},
        {"role": "user", "content": "second"},
    ]
    with tempfile.TemporaryDirectory() as tmp:
        with pytest.raises(RuntimeError) as err:
            _client(tmp, DIES_ON_REPLAY).chat.completions.create(model="sonnet", messages=history)
    text = str(err.value)
    assert text.startswith("Native exited before replay acknowledgment")
    assert "exit 3" in text and "replay frame rejected: bad content block" in text


def test_unknown_tool_is_passed_to_the_host_instead_of_discarding_the_reply(caplog):
    with tempfile.TemporaryDirectory() as tmp:
        with caplog.at_level("WARNING"):
            response = _client(tmp, UNKNOWN_TOOL).chat.completions.create(
                model="sonnet", messages=[{"role": "user", "content": "go"}], tools=[_tool("terminal")])
    message = response.choices[0].message
    assert message.content == "calling"
    assert [c.function.name for c in message.tool_calls] == ["process_manage"]
    assert response.choices[0].finish_reason == "tool_calls"
    assert "mcp__hermes__process_manage" in caplog.text


def test_create_under_a_running_loop_serves_sync_and_async_callers():
    request = dict(model="sonnet", messages=[{"role": "user", "content": "go"}])
    with tempfile.TemporaryDirectory() as tmp:
        client = _client(tmp, PLAIN)

        async def sync_caller_inside_a_task():
            # What Relay does: a synchronous provider callback executed on a loop thread.
            pending = client.chat.completions.create(**request)
            return pending.choices[0].message.content

        async def async_caller():
            return (await client.chat.completions.create(**request)).choices[0].message.content

        assert asyncio.run(sync_caller_inside_a_task()) == "hi"
        assert asyncio.run(async_caller()) == "hi"
        assert client.chat.completions.create(**request).choices[0].message.content == "hi"


def test_request_validation_errors_are_not_retried(profile):
    verdict = profile.classify_api_error(ValueError("Unsupported request parameters: extra_headers"))
    assert verdict == {"reason": "format_error", "retryable": False, "should_fallback": False}
    # Native/transport failures keep the host's default retry policy.
    assert profile.classify_api_error(RuntimeError("Native exited before replay acknowledgment")) is None
