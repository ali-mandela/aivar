"""Pytest configuration for generated flow tests."""

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
