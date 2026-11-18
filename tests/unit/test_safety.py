"""Unit tests for Safety Manager."""

import pytest
from control_panel.safety.manager import SafetyManager
from control_panel.safety.models import SafetyContext


def _ctx(
    worker_id: str = "w1",
    task_id: str = "t1",
    mode: str = "read_write",
    allowed: list[str] | None = None,
    dry_run: bool = False,
) -> SafetyContext:
    return SafetyContext(
        worker_id=worker_id,
        task_id=task_id,
        permission_mode=mode,
        allowed_paths=allowed or [],
        dry_run=dry_run,
    )


class TestSafetyManager:

    def setup_method(self) -> None:
        self.sm = SafetyManager(
            denylist=["C:\\Windows", "C:\\Program Files"],
            max_ops_per_second=100,
        )

    def test_valid_path_read_only(self, tmp_path) -> None:
        ctx = _ctx(mode="read_only")
        result = self.sm.validate_path(str(tmp_path), ctx, is_write=False)
        assert result.allowed
        assert result.sanitized_path is not None

    def test_denylist_blocks_access(self) -> None:
        ctx = _ctx()
        result = self.sm.validate_path("C:\\Windows\\System32", ctx)
        assert not result.allowed
        assert "denylist" in result.reason

    def test_read_only_blocks_write(self, tmp_path) -> None:
        ctx = _ctx(mode="read_only")
        result = self.sm.validate_path(str(tmp_path), ctx, is_write=True)
        assert not result.allowed
        assert "read_only" in result.reason

    def test_allowed_paths_restrict_access(self, tmp_path) -> None:
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()
        ctx = _ctx(allowed=[str(allowed_dir)])

        # Path inside allowed → OK
        ok = self.sm.validate_path(str(allowed_dir / "file.txt"), ctx)
        assert ok.allowed

        # Path outside allowed → blocked
        other = tmp_path / "other.txt"
        blocked = self.sm.validate_path(str(other), ctx)
        assert not blocked.allowed

    def test_forbidden_v01_tool(self) -> None:
        ctx = _ctx()
        result = self.sm.validate_tool("fs.delete_file", ctx)
        assert not result.allowed
        assert "forbidden" in result.reason.lower()

    def test_mutating_tool_blocked_for_read_only(self) -> None:
        ctx = _ctx(mode="read_only")
        result = self.sm.validate_tool("fs.move_file", ctx)
        assert not result.allowed

    def test_rate_limiting(self) -> None:
        sm = SafetyManager(max_ops_per_second=3)
        ctx = _ctx()
        for _ in range(3):
            r = sm.validate_tool("fs.list_directory", ctx)
            assert r.allowed
        # 4th should be blocked
        r = sm.validate_tool("fs.list_directory", ctx)
        assert not r.allowed

    def test_dry_run_flag(self) -> None:
        sm = SafetyManager(dry_run=True)
        ctx = _ctx()
        assert sm.is_dry_run(ctx)

    def test_context_dry_run_overrides(self) -> None:
        sm = SafetyManager(dry_run=False)
        ctx = _ctx(dry_run=True)
        assert sm.is_dry_run(ctx)
