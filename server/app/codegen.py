"""Code generation for compiled flows into executable pytest files.

This module renders Flow objects into pytest-executable Python modules, with
proper handling of selectors, value expressions, and secret placeholder resolution.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Callable

from app.contracts.contracts import Flow
from app.models.models import Selector, Step, StepKind
from app.secrets import PLACEHOLDER_RE


def locator_expr(selector: Selector) -> str:
    """Return the Playwright Python expression source for a selector.

    Matches build_locator's semantics exactly:
    - role   → page.get_by_role("<role>", name="<value>")
    - label  → page.get_by_label("<value>")
    - placeholder → page.get_by_placeholder("<value>")
    - text   → page.get_by_text("<value>")
    - testid → page.get_by_test_id("<value>")
    - css    → page.locator("<value>")

    Uses repr()-style safe quoting so values with quotes cannot break the source.
    """
    value_repr = repr(selector.value)

    if selector.strategy == "role":
        role_repr = repr(selector.role) if selector.role else "None"
        return f"page.get_by_role({role_repr}, name={value_repr})"
    elif selector.strategy == "label":
        return f"page.get_by_label({value_repr})"
    elif selector.strategy == "placeholder":
        return f"page.get_by_placeholder({value_repr})"
    elif selector.strategy == "text":
        return f"page.get_by_text({value_repr})"
    elif selector.strategy == "testid":
        return f"page.get_by_test_id({value_repr})"
    elif selector.strategy == "css":
        return f"page.locator({value_repr})"
    else:
        raise ValueError(f"Unknown selector strategy: {selector.strategy}")


def sanitize_identifier(name: str) -> str:
    """Sanitize a string into a valid Python identifier.

    - Lowercase all characters
    - Replace non-alphanumerics with underscores
    - Collapse consecutive underscores
    - Strip leading/trailing underscores
    - Prefix with 't_' if it starts with a digit
    - Fall back to 'flow' if empty
    """
    # Lowercase
    name = name.lower()

    # Replace non-alphanumerics with underscores
    name = re.sub(r"[^a-z0-9_]", "_", name)

    # Collapse consecutive underscores
    name = re.sub(r"_+", "_", name)

    # Strip leading/trailing underscores
    name = name.strip("_")

    # If empty, return fallback
    if not name:
        return "flow"

    # If starts with digit, prefix with 't_'
    if name[0].isdigit():
        name = "t_" + name

    return name


def _value_expr(value: str | None) -> str:
    """Generate a Python expression for a step value.

    - None → ""
    - ${NAME} → os.environ["NAME"]
    - ${NAME:-default} → os.environ.get("NAME", "default")
    - Plain literals → quoted string (repr style)

    Never inlines a secret.
    """
    if value is None:
        return '""'

    # Check if value contains any placeholders
    matches = list(PLACEHOLDER_RE.finditer(value))

    if not matches:
        # No placeholders, just a literal string
        return repr(value)

    # If there's exactly one match and it spans the entire value, emit the env lookup
    if len(matches) == 1 and matches[0].start() == 0 and matches[0].end() == len(value):
        match = matches[0]
        name = match.group(1)
        default = match.group(2)

        if default is not None:
            return f'os.environ.get({repr(name)}, {repr(default)})'
        else:
            return f'os.environ[{repr(name)}]'

    # Multiple placeholders or inline placeholders: need string formatting
    # Replace all placeholders with f-string expressions
    result = value
    for match in reversed(matches):
        name = match.group(1)
        default = match.group(2)

        if default is not None:
            replacement = f'{{os.environ.get({repr(name)}, {repr(default)})}}'
        else:
            replacement = f'{{os.environ[{repr(name)}]}}'

        start, end = match.span()
        result = result[:start] + replacement + result[end:]

    # Now wrap the whole thing in an f-string
    return f"f{repr(result)}"


def render_pytest(
    flow: Flow,
    url: str,
    *,
    username_env: str = "AIVAR_USERNAME",
    password_env: str = "AIVAR_PASSWORD",
) -> str:
    """Produce a complete pytest module from a compiled flow.

    Requirements:
    - Header comment with generation info
    - import os and from playwright.sync_api import Page, expect
    - One test function named test_<sanitized flow name>(page: Page):
    - First line: page.goto("<url>") using flow.entry_url or url
    - Per step with proper locator expressions and value handling
    - All secrets properly resolved via environment variables
    - Each step preceded by a comment
    - Ends with newline
    """
    if not flow.is_compiled:
        raise ValueError(f"Flow {flow.id} is not compiled (missing selectors)")

    sanitized_name = sanitize_identifier(flow.name)
    entry_url = flow.entry_url or url

    lines = []

    # Header comment
    lines.append("# Generated by the aivar orchestration agent")
    lines.append(f"# Flow ID: {flow.id}")
    lines.append(f"# Flow name: {flow.name}")
    lines.append("# Regenerate rather than edit by hand")
    lines.append("")

    # Imports
    lines.append("import os")
    lines.append("from playwright.sync_api import Page, expect")
    lines.append("")

    # Test function
    lines.append(f"def test_{sanitized_name}(page: Page):")
    lines.append(f'    page.goto({repr(entry_url)})')

    # Process steps
    for step in flow.steps:
        # assert_url is about the page, not an element, so it has no locator to
        # render -- and a compiled step for it may carry no selector at all.
        locator = locator_expr(step.selector) if step.selector else None
        kind_desc = f"{step.kind.value}: {step.target}"
        lines.append(f"    # {kind_desc}")

        if step.verb == "click":
            lines.append(f"    {locator}.first.click()")
        elif step.verb == "fill":
            value_expr_str = _value_expr(step.value)
            lines.append(f"    {locator}.first.fill({value_expr_str})")
        elif step.verb == "select":
            lines.append(f"    {locator}.first.select_option(label={_value_expr(step.value)})")
        elif step.verb == "check":
            lines.append(f"    {locator}.first.check()")
        elif step.verb == "press":
            lines.append(f"    {locator}.first.press({_value_expr(step.value or 'Enter')})")
        elif step.verb == "wait_visible":
            if step.kind == StepKind.ASSERTION:
                lines.append(f"    expect({locator}.first).to_be_visible()")
            else:
                lines.append(f"    {locator}.first.wait_for(state=\"visible\")")
        elif step.verb == "assert_text":
            # to_contain_text, not to_have_text: a person reading an error
            # inside a longer banner considers the check met. expect() also
            # retries, so this rides out a message that renders a beat late.
            lines.append(f"    expect({locator}.first).to_contain_text({_value_expr(step.value)})")
        elif step.verb == "assert_url":
            # Glob rather than a regex: it needs no import in the generated
            # file, and wait_for_url retries, so an SPA that swaps the address
            # a moment after the click still passes.
            lines.append(f"    page.wait_for_url({repr('**' + (step.value or '') + '**')})")
        elif step.verb == "assert_hidden":
            lines.append(f"    expect({locator}.first).to_be_hidden()")
        else:
            raise ValueError(f"Unknown verb: {step.verb}")

    # End with newline
    lines.append("")

    return "\n".join(lines)


def write_flow_file(
    flow: Flow,
    url: str,
    out_dir: str | Path = "tests/generated",
) -> Path:
    """Write a compiled flow as a pytest file.

    File is named test_<sanitized flow name>.py and placed in out_dir.
    Directory is created if it doesn't exist.
    Returns the path to the written file.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    sanitized_name = sanitize_identifier(flow.name)
    file_path = out_path / f"test_{sanitized_name}.py"

    content = render_pytest(flow, url)
    file_path.write_text(content, encoding="utf-8")

    return file_path


def write_suite(
    flows: list[Flow],
    url: str,
    out_dir: str | Path = "tests/generated",
) -> list[Path]:
    """Write all compiled flows as pytest files and ensure conftest.py exists.

    Skips uncompiled flows (does not crash).
    Returns list of paths written.
    Also writes a minimal conftest.py if absent.
    """
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    written_paths = []

    for flow in flows:
        if flow.is_compiled:
            path = write_flow_file(flow, url, out_path)
            written_paths.append(path)

    # Write conftest.py if it doesn't exist
    conftest_path = out_path / "conftest.py"
    if not conftest_path.exists():
        conftest_content = '''"""Pytest configuration for generated flow tests."""

import pytest
from playwright.sync_api import sync_playwright, Page


@pytest.fixture
def browser_context_args():
    """Configure browser context to ignore HTTPS errors."""
    return {
        "ignore_https_errors": True,
    }


@pytest.fixture
def page() -> Page:
    """Provide a Playwright page for tests."""
    with sync_playwright() as p:
        browser = p.chromium.launch()
        context = browser.new_context(**{"ignore_https_errors": True})
        page = context.new_page()
        try:
            yield page
        finally:
            context.close()
            browser.close()
'''
        conftest_path.write_text(conftest_content, encoding="utf-8")

    return written_paths


def is_importable(path: str | Path) -> tuple[bool, str]:
    """Validate that a file is importable Python.

    Returns (True, "") on success or (False, <error message>) on syntax error.
    """
    path = Path(path)

    try:
        source = path.read_text(encoding="utf-8")
        ast.parse(source)
        return (True, "")
    except SyntaxError as e:
        return (False, str(e))
    except Exception as e:
        return (False, f"Error reading file: {str(e)}")
