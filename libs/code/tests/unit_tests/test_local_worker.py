from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from deepagents_code._cli_context import CLIContextSchema
from deepagents_code.client.launch.server import generate_langgraph_json


def test_registration_only_for_builtin_graph(tmp_path):
    import json

    path = generate_langgraph_json(tmp_path)
    assert json.loads(path.read_text())["graphs"]["local_worker"] == (
        "deepagents_code.server_graph:make_worker_graph"
    )
    path = generate_langgraph_json(tmp_path, graph_ref="custom:graph")
    assert json.loads(path.read_text())["graphs"] == {"agent": "custom:graph"}


async def test_worker_binds_child_from_validated_parent():
    from deepagents_code import server_graph

    worker = object()
    binding = SimpleNamespace(
        cwd="/workspace",
        config_fingerprint="trusted",
        workspace_config=lambda: {"policy": "trusted"},
    )
    context = CLIContextSchema(
        workspace={"workspace_id": "workspace"}, thread_id="parent"
    )
    runtime = SimpleNamespace(execution_runtime=SimpleNamespace(context=context))
    config = {
        "configurable": {"thread_id": "child"},
        "metadata": {"parent_thread_id": "parent"},
    }
    with (
        patch(
            "deepagents_code.workspace.require_thread_workspace",
            new=AsyncMock(return_value=binding),
        ) as require,
        patch(
            "deepagents_code.workspace.bind_thread_workspace",
            new=AsyncMock(return_value=binding),
        ) as bind,
        patch.object(
            server_graph,
            "_workspace_runtime",
            new=AsyncMock(return_value=SimpleNamespace(worker=worker)),
        ),
    ):
        assert await server_graph.make_worker_graph(config, runtime) is worker
    assert require.await_args_list[0].args == ("parent", context.workspace)
    assert require.await_args_list[1].args == ("child", context.workspace)
    bind.assert_awaited_once_with(
        "child", "/workspace", {"policy": "trusted"}, config_fingerprint="trusted"
    )


async def test_worker_rejects_missing_parent_before_binding():
    from deepagents_code import server_graph

    runtime = SimpleNamespace(
        execution_runtime=SimpleNamespace(
            context=CLIContextSchema(workspace={"workspace_id": "workspace"})
        )
    )
    with (
        patch(
            "deepagents_code.workspace.bind_thread_workspace", new=AsyncMock()
        ) as bind,
        pytest.raises(ValueError, match="distinct parent"),
    ):
        await server_graph.make_worker_graph(
            {"configurable": {"thread_id": "child"}}, runtime
        )
    bind.assert_not_awaited()
