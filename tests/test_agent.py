"""Tests unitarios de agent.py. Sin red. Cliente OpenAI falso."""

import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import agent  # noqa: E402


def _message(*, content=None, tool_calls=None):
    data = {
        "role": "assistant",
        "content": content,
    }
    if tool_calls is not None:
        data["tool_calls"] = tool_calls

    def model_dump(exclude_none=False):
        if exclude_none:
            return {k: v for k, v in data.items() if v is not None}
        return dict(data)

    return SimpleNamespace(
        content=content,
        tool_calls=tool_calls,
        model_dump=model_dump,
    )


def _tool_call(call_id: str, name: str, arguments: str):
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _completion(finish_reason: str, message):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason=finish_reason, message=message)]
    )


class FakeCompletions:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = 0

    def create(self, **kwargs):
        self.calls += 1
        if not self._responses:
            raise AssertionError("FakeCompletions: no more responses")
        return self._responses.pop(0)


class FakeClient:
    def __init__(self, responses):
        self.chat = SimpleNamespace(completions=FakeCompletions(responses))


class AgentPathTests(unittest.TestCase):
    def setUp(self):
        self._tmpdir = tempfile.TemporaryDirectory()
        self.root = Path(self._tmpdir.name).resolve()
        (self.root / "workspace").mkdir()
        (self.root / "workspace" / "notes.txt").write_text("hello\n", encoding="utf-8")
        (self.root / ".env").write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")
        self._work_root = agent.WORK_ROOT
        agent.WORK_ROOT = self.root

    def tearDown(self):
        agent.WORK_ROOT = self._work_root
        self._tmpdir.cleanup()

    def test_missing_file_returns_text(self):
        result = agent.read_file("workspace/missing.txt")
        self.assertIn("Error leyendo", result)
        self.assertIn("missing.txt", result)

    def test_parent_escape_rejected(self):
        result = agent.read_file("../outside.txt")
        self.assertIn("sale del directorio", result)

    def test_outside_symlink_rejected(self):
        outside = Path(self._tmpdir.name).parent / f"outside-{os.getpid()}.txt"
        try:
            outside.write_text("leak\n", encoding="utf-8")
            link = self.root / "workspace" / "leak.txt"
            link.symlink_to(outside)
            result = agent.read_file("workspace/leak.txt")
            self.assertIn("sale del directorio", result)
            # El contenido del archivo externo nunca debe devolverse.
            self.assertNotIn("leak\n", result)
            self.assertNotEqual(result.strip(), "leak")
        finally:
            if outside.exists():
                outside.unlink()

    def test_env_read_refused(self):
        result = agent.read_file(".env")
        self.assertIn("Rechazado", result)
        self.assertNotIn("secret", result)

    def test_env_write_refused(self):
        result = agent.write_file(".env", "OPENAI_API_KEY=hacked\n")
        self.assertIn("Rechazado", result)
        self.assertEqual(
            (self.root / ".env").read_text(encoding="utf-8"),
            "OPENAI_API_KEY=secret\n",
        )

    def test_truncate_output(self):
        long_text = "x" * (agent.MAX_TOOL_OUTPUT + 50)
        clipped = agent.truncate_output(long_text)
        self.assertIn("output truncated", clipped)
        self.assertLess(len(clipped), len(long_text))

    def test_unknown_tool(self):
        result = agent.execute_tool("nope", {}, approve=lambda _: True)
        self.assertIn("Unknown tool", result)

    def test_bad_json_arguments_in_loop(self):
        tool_calls = [_tool_call("c1", "read_file", "{not-json")]
        responses = [
            _completion("tool_calls", _message(tool_calls=tool_calls)),
            _completion("stop", _message(content="Handled.")),
        ]
        client = FakeClient(responses)
        messages: list[dict] = [{"role": "user", "content": "read"}]
        answer = agent.run_agent(client, messages, approve=lambda _: True)
        self.assertEqual(answer, "Handled.")
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertIn("JSON inválido", tool_msgs[0]["content"])

    def test_rejected_write_does_not_create_file(self):
        target = self.root / "workspace" / "new.txt"
        result = agent.execute_tool(
            "write_file",
            {"path": "workspace/new.txt", "content": "nope"},
            approve=lambda _: False,
        )
        self.assertEqual(result, "GUARDRAIL: Usuario rechazó esta acción.")
        self.assertFalse(target.exists())

    def test_approved_write_stays_inside(self):
        result = agent.execute_tool(
            "write_file",
            {"path": "workspace/ok.txt", "content": "ok"},
            approve=lambda _: True,
        )
        self.assertIn("Escrito", result)
        self.assertEqual(
            (self.root / "workspace" / "ok.txt").read_text(encoding="utf-8"),
            "ok",
        )

    def test_command_timeout_returns_text(self):
        with patch.object(agent, "COMMAND_TIMEOUT_SECONDS", 0.1):
            result = agent.run_command("sleep 2")
        self.assertIn("timed out", result)

    def test_three_tool_calls_produce_three_tool_messages(self):
        tool_calls = [
            _tool_call("a", "read_file", '{"path":"workspace/notes.txt"}'),
            _tool_call("b", "list_files", '{"path":"workspace"}'),
            _tool_call("c", "list_files", '{"path":"."}'),
        ]
        responses = [
            _completion("tool_calls", _message(tool_calls=tool_calls)),
            _completion("stop", _message(content="Done.")),
        ]
        client = FakeClient(responses)
        messages: list[dict] = [{"role": "user", "content": "read three"}]
        answer = agent.run_agent(client, messages, approve=lambda _: True)
        self.assertEqual(answer, "Done.")
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 3)
        self.assertEqual([m["tool_call_id"] for m in tool_msgs], ["a", "b", "c"])

    def test_finish_reason_stop_ends(self):
        client = FakeClient([_completion("stop", _message(content="Hello."))])
        messages: list[dict] = [{"role": "user", "content": "hi"}]
        answer = agent.run_agent(client, messages)
        self.assertEqual(answer, "Hello.")
        self.assertEqual(client.chat.completions.calls, 1)

    def test_finish_reason_length_not_success(self):
        client = FakeClient([_completion("length", _message(content="cut"))])
        messages: list[dict] = [{"role": "user", "content": "hi"}]
        answer = agent.run_agent(client, messages)
        self.assertIn("límite de tokens", answer)

    def test_iteration_cap_stops(self):
        tool_calls = [_tool_call("x", "list_files", '{"path":"."}')]
        # Siempre pide tools otra vez: el bucle debe chocar con el tope.
        responses = [
            _completion("tool_calls", _message(tool_calls=tool_calls)) for _ in range(5)
        ]
        client = FakeClient(responses)
        messages: list[dict] = [{"role": "user", "content": "loop"}]
        answer = agent.run_agent(
            client,
            messages,
            approve=lambda _: True,
            max_iterations=3,
        )
        self.assertIn("tope de iteraciones", answer)
        self.assertEqual(client.chat.completions.calls, 3)

    def test_rejected_command_still_sends_tool_message(self):
        tool_calls = [_tool_call("r1", "run_command", '{"command":"echo hi"}')]
        responses = [
            _completion("tool_calls", _message(tool_calls=tool_calls)),
            _completion("stop", _message(content="Understood.")),
        ]
        client = FakeClient(responses)
        messages: list[dict] = [{"role": "user", "content": "run"}]
        answer = agent.run_agent(client, messages, approve=lambda _: False)
        self.assertEqual(answer, "Understood.")
        tool_msgs = [m for m in messages if m.get("role") == "tool"]
        self.assertEqual(len(tool_msgs), 1)
        self.assertEqual(tool_msgs[0]["content"], "GUARDRAIL: Usuario rechazó esta acción.")
        self.assertEqual(tool_msgs[0]["tool_call_id"], "r1")


if __name__ == "__main__":
    unittest.main()
