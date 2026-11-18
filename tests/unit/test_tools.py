"""Unit tests for filesystem tools."""

import pytest
from pathlib import Path

from control_panel.safety.models import SafetyContext
from control_panel.tools.filesystem import (
    ListDirectoryTool,
    GetMetadataTool,
    CreateDirectoryTool,
    MoveFileTool,
    CopyFileTool,
)


def _ctx(
    mode: str = "read_write",
    allowed: list[str] | None = None,
    dry_run: bool = False,
) -> SafetyContext:
    return SafetyContext(
        worker_id="w1",
        task_id="t1",
        permission_mode=mode,
        allowed_paths=allowed or [],
        dry_run=dry_run,
    )


class TestListDirectory:
    def test_lists_files(self, tmp_path) -> None:
        (tmp_path / "a.txt").write_text("a")
        (tmp_path / "b.txt").write_text("b")
        tool = ListDirectoryTool()
        result = tool.execute({"path": str(tmp_path)}, _ctx())
        assert result.success
        assert result.output["count"] == 2

    def test_nonexistent_path(self) -> None:
        tool = ListDirectoryTool()
        result = tool.execute({"path": "C:/nonexistent/path/xyz"}, _ctx())
        assert not result.success

    def test_file_not_directory(self, tmp_path) -> None:
        f = tmp_path / "file.txt"
        f.write_text("content")
        tool = ListDirectoryTool()
        result = tool.execute({"path": str(f)}, _ctx())
        assert not result.success


class TestGetMetadata:
    def test_gets_file_metadata(self, tmp_path) -> None:
        f = tmp_path / "test.pdf"
        f.write_bytes(b"x" * 100)
        tool = GetMetadataTool()
        result = tool.execute({"path": str(f)}, _ctx())
        assert result.success
        assert result.output["name"] == "test.pdf"
        assert result.output["size_bytes"] == 100
        assert result.output["suffix"] == ".pdf"
        assert result.output["is_file"]


class TestCreateDirectory:
    def test_creates_directory(self, tmp_path) -> None:
        new_dir = tmp_path / "new" / "sub"
        tool = CreateDirectoryTool()
        result = tool.execute({"path": str(new_dir)}, _ctx())
        assert result.success
        assert new_dir.exists()
        assert len(result.side_effects) == 1

    def test_dry_run_does_not_create(self, tmp_path) -> None:
        new_dir = tmp_path / "would_create"
        tool = CreateDirectoryTool()
        result = tool.execute({"path": str(new_dir)}, _ctx(dry_run=True))
        assert result.success
        assert not new_dir.exists()
        assert result.output["dry_run"]

    def test_read_only_blocked(self, tmp_path) -> None:
        tool = CreateDirectoryTool()
        result = tool.execute({"path": str(tmp_path / "d")}, _ctx(mode="read_only"))
        assert not result.success


class TestMoveFile:
    def test_moves_file(self, tmp_path) -> None:
        src = tmp_path / "a.txt"
        dst = tmp_path / "moved" / "a.txt"
        src.write_text("content")
        tool = MoveFileTool()
        result = tool.execute({"source": str(src), "destination": str(dst)}, _ctx())
        assert result.success
        assert dst.exists()
        assert not src.exists()
        assert len(result.side_effects) == 1
        assert result.side_effects[0].type == "file_move"

    def test_source_not_found(self, tmp_path) -> None:
        tool = MoveFileTool()
        result = tool.execute(
            {"source": str(tmp_path / "ghost.txt"), "destination": str(tmp_path / "out.txt")},
            _ctx()
        )
        assert not result.success

    def test_dry_run_no_move(self, tmp_path) -> None:
        src = tmp_path / "src.txt"
        src.write_text("hi")
        dst = tmp_path / "dst.txt"
        tool = MoveFileTool()
        result = tool.execute({"source": str(src), "destination": str(dst)}, _ctx(dry_run=True))
        assert result.success
        assert src.exists()
        assert not dst.exists()
        assert result.output["dry_run"]


class TestCopyFile:
    def test_copies_file(self, tmp_path) -> None:
        src = tmp_path / "orig.txt"
        dst = tmp_path / "copy.txt"
        src.write_text("data")
        tool = CopyFileTool()
        result = tool.execute({"source": str(src), "destination": str(dst)}, _ctx())
        assert result.success
        assert src.exists()  # original preserved
        assert dst.exists()
        assert dst.read_text() == "data"
