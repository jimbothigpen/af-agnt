import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import pytest

from deepagents_code import server_graph
from deepagents_code._cli_context import CLIContextSchema
from deepagents_code._env_vars import SERVER_ENV_PREFIX
from deepagents_code.thread_ownership import (
    OWNER_KEY,
    ThreadOwnershipError,
    ensure_owned,
    held_lease,
    writer_guard,
)


@pytest.fixture
async def worker_owner(tmp_path, monkeypatch):
    db_path = tmp_path / "server.db"
    monkeypatch.setenv(f"{SERVER_ENV_PREFIX}DB_PATH", str(db_path))
    monkeypatch.setattr(server_graph, "_worker_threads", {})
    monkeypatch.setattr(server_graph, "_workers_stopping", False)
    parent = await asyncio.to_thread(ensure_owned, "parent", db_path=db_path)
    runtime = SimpleNamespace(
        config={"configurable": {"thread_id": "parent", OWNER_KEY: parent.token}},
        context=CLIContextSchema(workspace={"workspace_id": "workspace"}),
    )
    with patch("deepagents_code.workspace.require_thread_workspace", new=AsyncMock()):
        try:
            yield runtime, db_path
        finally:
            for _, lease in tuple(server_graph._worker_threads.values()):
                await asyncio.to_thread(lease.release)
            server_graph._worker_threads.clear()
            await asyncio.to_thread(parent.release)


async def test_worker_reservation_and_update_reuse(worker_owner):
    runtime, db_path = worker_owner
    async with server_graph.prepare_local_worker_run("child", runtime) as options:
        token = options["headers"][OWNER_KEY]
        async with writer_guard("child", db_path=db_path, token=token):
            pass
    lease = await asyncio.to_thread(held_lease, "child", db_path=db_path)
    assert lease.token == token
    async with server_graph.prepare_local_worker_run("child", runtime) as options:
        assert options["headers"][OWNER_KEY] == token


async def test_failed_launch_releases_only_new_reservation(worker_owner):
    runtime, db_path = worker_owner
    message = "launch failed"
    with pytest.raises(RuntimeError, match="launch failed"):
        async with server_graph.prepare_local_worker_run("child", runtime):
            raise RuntimeError(message)
    assert await asyncio.to_thread(held_lease, "child", db_path=db_path) is None
    assert not server_graph._worker_threads
    async with server_graph.prepare_local_worker_run("child", runtime):
        pass
    message = "update failed"
    with pytest.raises(RuntimeError, match="update failed"):
        async with server_graph.prepare_local_worker_run("child", runtime):
            raise RuntimeError(message)
    assert await asyncio.to_thread(held_lease, "child", db_path=db_path) is not None


async def test_stale_parent_cannot_reserve_child(worker_owner):
    runtime, _ = worker_owner
    runtime.config["configurable"][OWNER_KEY] = "stale"
    with pytest.raises(ThreadOwnershipError):
        async with server_graph.prepare_local_worker_run("child", runtime):
            pytest.fail("Stale parent launched a worker")
    assert not server_graph._worker_threads


async def test_workspace_conflict_prevents_reservation(worker_owner):
    runtime, _ = worker_owner
    with (
        patch(
            "deepagents_code.workspace.require_thread_workspace",
            new=AsyncMock(side_effect=ValueError("workspace mismatch")),
        ),
        pytest.raises(ValueError, match="workspace mismatch"),
    ):
        async with server_graph.prepare_local_worker_run("child", runtime):
            pytest.fail("Invalid workspace launched a worker")
    assert not server_graph._worker_threads


async def test_worker_rejects_another_parent(worker_owner):
    runtime, _ = worker_owner
    async with server_graph.prepare_local_worker_run("child", runtime):
        pass
    runtime.config["configurable"]["thread_id"] = "other"
    with pytest.raises(ValueError, match="another parent"):
        async with server_graph.prepare_local_worker_run("child", runtime):
            pytest.fail("Worker changed parents")


async def test_shutdown_cancels_before_releasing(worker_owner):
    runtime, _ = worker_owner
    async with server_graph.prepare_local_worker_run("child", runtime):
        pass
    lease = server_graph._worker_threads["child"][1]
    cancel = AsyncMock()
    release = Mock(wraps=lease.release)
    client = SimpleNamespace(
        runs=SimpleNamespace(
            list=AsyncMock(
                return_value=[
                    {"status": "running", "run_id": "active"},
                    {"status": "success", "run_id": "done"},
                ]
            ),
            cancel=cancel,
        )
    )

    def cancelled(*_args: object, **_kwargs: object) -> None:
        release.assert_not_called()

    cancel.side_effect = cancelled
    with (
        patch("langgraph_sdk.get_client", return_value=client),
        patch.object(lease, "release", release),
    ):
        await server_graph.shutdown_local_workers()
    cancel.assert_awaited_once_with("child", "active", wait=True)
    release.assert_called_once_with()
    assert not server_graph._worker_threads
    with pytest.raises(RuntimeError, match="shutting down"):
        async with server_graph.prepare_local_worker_run("new", runtime):
            pytest.fail("Launched during shutdown")


async def test_shutdown_failure_keeps_lease_for_active_run(worker_owner):
    runtime, _ = worker_owner
    async with server_graph.prepare_local_worker_run("child", runtime):
        pass
    lease = server_graph._worker_threads["child"][1]
    client = SimpleNamespace(
        runs=SimpleNamespace(list=AsyncMock(side_effect=RuntimeError("unavailable")))
    )
    with (
        patch("langgraph_sdk.get_client", return_value=client),
        patch.object(lease, "release") as release,
    ):
        await server_graph.shutdown_local_workers()
    release.assert_not_called()
    assert "child" in server_graph._worker_threads
