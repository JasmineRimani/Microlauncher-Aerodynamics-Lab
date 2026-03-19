from __future__ import annotations

from pathlib import Path

import pytest


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register optional reporting outputs for the test suite."""
    parser.addoption(
        "--report-dir",
        action="store",
        default=None,
        help="Directory where optional test plots and tables will be written.",
    )


@pytest.fixture
def report_dir(pytestconfig: pytest.Config) -> Path | None:
    """Return the requested report directory, creating it on demand."""
    report_dir_value = pytestconfig.getoption("report_dir")
    if not report_dir_value:
        return None

    path = Path(report_dir_value).resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path
