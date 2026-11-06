import asyncio
import sys
import uuid
from pathlib import Path
from typing import Any

import pytest
from langchain_core.messages import AIMessage
from langgraph.types import Command, StateSnapshot

from deepagents_code._testing_models import (
    DCA_TEST_WRITE_FILE_MARKER,
    TOP_LEVEL_WRITE_CONTENT,
)


@pytest.mark.timeout(180)
@pytest.mark.parametrize("auto_approve", [True, False], ids=["auto", "manual"])
async def test_registered_worker_launches_through_asgi(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, *, auto_approve: bool
) -> None:
    from deepagents_code import model_config
    from deepagents_code._paths import _capture_paths
    from deepagents_code.client.launch.server_manager import server_session
    from deepagents_code.config import _ensure_bootstrap, create_model

    await asyncio.to_thread(_ensure_bootstrap)

    home = tmp_path / "home"
    project = tmp_path / "project"
    profile = home / ".deepagents"
    profile.mkdir(parents=True)
    project.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("DEEPAGENTS_HOME", str(profile))
    monkeypatch.setenv("DEEPAGENTS_CODE_NO_UPDATE_CHECK", "1")
    monkeypatch.chdir(project)
    paths = _capture_paths(str(profile), launch_home=home)
    for name, module in list(sys.modules.items()):
        if name.startswith("deepagents_code") and hasattr(module, "PATHS"):
            monkeypatch.setattr(module, "PATHS", paths)
    config_path = profile / "config.toml"
    config_path.write_text(
        "[models.providers.itest]\n"
        'class_path = "deepagents_code._testing_models:'
        'ToolCallingIntegrationChatModel"\n'
        'models = ["fake"]\n'
        "[models.providers.itest.profile]\n"
        "max_input_tokens = 32000\n"
    )
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_DIR", profile)
    monkeypatch.setattr(model_config, "DEFAULT_CONFIG_PATH", config_path)
    model_config.clear_caches()
    try:
        create_model("itest:fake").apply_to_runtime_state()
        async with server_session(
            assistant_id="worker-test",
            model_name="itest:fake",
            no_mcp=True,
            enable_shell=False,
            enable_interpreter=False,
            auto_approve=auto_approve,
            interactive=True,
            sandbox_type="none",
        ) as (agent, server):
            parent = str(uuid.uuid4())
            config = {"configurable": {"thread_id": parent}}
            context = {"thread_id": parent, "auto_approve": auto_approve}
            target = project / "worker-result.txt"
            await agent.aensure_thread(config)
            await agent.abind_workspace(config, cwd=str(project))

            async def call_parent_tool(
                name: str, args: dict[str, Any], call_id: str
            ) -> StateSnapshot:
                await agent.aupdate_state(
                    config,
                    {
                        "messages": [
                            AIMessage(
                                content="",
                                tool_calls=[
                                    {"name": name, "args": args, "id": call_id}
                                ],
                            )
                        ]
                    },
                    as_node="model",
                )
                async for _ in agent.astream(
                    None,
                    config=config,
                    stream_mode=["updates"],
                    subgraphs=True,
                    context=context,
                ):
                    pass
                state = await agent.aget_state(config)
                if state.interrupts:
                    decisions = {
                        item.id: {"decisions": [{"type": "approve"}]}
                        for item in state.interrupts
                    }
                    async for _ in agent.astream(
                        Command(resume=decisions),
                        config=config,
                        stream_mode=["updates"],
                        subgraphs=True,
                        context=context,
                    ):
                        pass
                return await agent.aget_state(config)

            state = await call_parent_tool(
                "start_async_task",
                {
                    "description": f"{DCA_TEST_WRITE_FILE_MARKER}{target}",
                    "subagent_type": "local_worker",
                },
                "launch-worker",
            )
            tasks = state.values.get("async_tasks", {})
            assert len(tasks) == 1, repr(state.values)
            task_id, task = next(iter(tasks.items()))
            assert task["thread_id"] != parent
            client = agent._get_graph()._validate_client()

            async def wait_for_run(tracked: dict[str, Any]) -> dict[str, Any]:
                for _ in range(200):
                    run = await client.runs.get(tracked["thread_id"], tracked["run_id"])
                    if run["status"] not in {"pending", "running"}:
                        return run
                    await asyncio.sleep(0.1)
                pytest.fail(f"Worker did not finish: {run!r}")

            run = await wait_for_run(task)
            assert run["status"] == "success", await asyncio.to_thread(
                Path(server._log_file.name).read_text, encoding="utf-8"
            )
            child = await client.threads.get(task["thread_id"])
            assert child["metadata"]["parent_thread_id"] == parent
            assert "parent_assistant_id" not in child["metadata"]
            child_state = await client.threads.get_state(task["thread_id"])
            assert child_state["values"]["messages"]
            assert "async_tasks" not in child_state["values"]
            if not auto_approve:
                interrupts = [
                    interrupt
                    for pending in child_state["tasks"]
                    for interrupt in pending.get("interrupts", [])
                ]
                assert len(interrupts) == 1, child_state
                requests = interrupts[0]["value"]["action_requests"]
                assert [request["name"] for request in requests] == ["write_file"]
                assert requests[0]["args"]["file_path"] == str(target)
                assert not target.exists()
                return

            assert target.read_text() == TOP_LEVEL_WRITE_CONTENT
            assert not child_state["tasks"]
            updated_message = "Worker follow-up: acknowledge update-marker-42"
            updated = await call_parent_tool(
                "update_async_task",
                {"task_id": task_id, "message": updated_message},
                "update-worker",
            )
            updated_tasks = updated.values["async_tasks"]
            assert set(updated_tasks) == {task_id}
            updated_task = updated_tasks[task_id]
            assert updated_task["thread_id"] == task["thread_id"]
            assert updated_task["run_id"] != task["run_id"]
            updated_run = await wait_for_run(updated_task)
            assert updated_run["status"] == "success", await asyncio.to_thread(
                Path(server._log_file.name).read_text, encoding="utf-8"
            )
            updated_child = await client.threads.get_state(task["thread_id"])
            messages = updated_child["values"]["messages"]
            assert any(
                message["type"] == "human" and message["content"] == updated_message
                for message in messages
            )
            assert any(message["type"] == "tool" for message in messages)
            assert messages[-1]["type"] == "ai"
            assert messages[-1]["content"] == "done"
            assert target.read_text() == TOP_LEVEL_WRITE_CONTENT
    finally:
        model_config.clear_caches()
