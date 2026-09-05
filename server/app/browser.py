from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.models.models import Selector


class SelectorConfigError(Exception):
    """Raised for a malformed Selector (a configuration bug, not an app bug)."""

    pass


def build_locator(page: Any, selector: Selector) -> Any:
    """Map a Selector to a Playwright locator based on strategy."""
    if selector.strategy == "role":
        if selector.role is None:
            raise SelectorConfigError(
                f"role strategy requires a role field, got None for selector value '{selector.value}'"
            )
        return page.get_by_role(selector.role, name=selector.value)
    elif selector.strategy == "label":
        return page.get_by_label(selector.value)
    elif selector.strategy == "placeholder":
        return page.get_by_placeholder(selector.value)
    elif selector.strategy == "text":
        return page.get_by_text(selector.value)
    elif selector.strategy == "testid":
        return page.get_by_test_id(selector.value)
    elif selector.strategy == "css":
        return page.locator(selector.value)
    else:
        raise SelectorConfigError(
            f"Unknown selector strategy '{selector.strategy}'"
        )


@dataclass(frozen=True)
class Node:
    """A snapshot of an accessible element on the page."""

    ref: str
    role: str
    name: str
    tag: str
    placeholder: str | None
    testid: str | None
    visible: bool
    testid_attr: str | None = None


class Browser:
    """Wrapper around a Playwright Page for snapshot and action methods."""

    def __init__(self, page: Any) -> None:
        self._page = page

    def snapshot(self) -> list[Node]:
        """
        Take a snapshot of the page and return accessible nodes.
        Uses the JS defined in the spec.
        """
        js_result = self._page.evaluate(
            """() => {
  const SEL = 'input,button,a,select,textarea,[role],[data-testid],[data-test],h1,h2,h3,h4,h5,h6,label';
  const implicitRole = (el) => {
    const t = el.tagName.toLowerCase();
    if (t === 'button') return 'button';
    if (t === 'a') return 'link';
    if (t === 'select') return 'combobox';
    if (t === 'textarea') return 'textbox';
    if (/^h[1-6]$/.test(t)) return 'heading';
    if (t === 'label') return 'label';
    if (t === 'input') {
      const ty = (el.getAttribute('type') || 'text').toLowerCase();
      if (ty === 'submit' || ty === 'button' || ty === 'reset') return 'button';
      if (ty === 'checkbox') return 'checkbox';
      if (ty === 'radio') return 'radio';
      return 'textbox';
    }
    return '';
  };
  const nameOf = (el) => {
    const aria = el.getAttribute('aria-label');
    if (aria && aria.trim()) return aria.trim();
    const id = el.getAttribute('id');
    if (id) {
      const lab = document.querySelector('label[for="' + CSS.escape(id) + '"]');
      if (lab && lab.textContent.trim()) return lab.textContent.trim();
    }
    const ph = el.getAttribute('placeholder');
    if (ph && ph.trim()) return ph.trim();
    const val = el.getAttribute('value');
    if (val && val.trim()) return val.trim();
    const txt = (el.textContent || '').trim();
    return txt.length > 0 && txt.length <= 120 ? txt : '';
  };
  return Array.from(document.querySelectorAll(SEL)).map((el) => ({
    role: el.getAttribute('role') || implicitRole(el),
    name: nameOf(el),
    tag: el.tagName.toLowerCase(),
    placeholder: el.getAttribute('placeholder'),
    testid: (el.getAttribute('data-testid') || el.getAttribute('data-test') || el.getAttribute('data-test-id')),
    testid_attr: (el.hasAttribute('data-testid') ? 'data-testid'
                : el.hasAttribute('data-test') ? 'data-test'
                : el.hasAttribute('data-test-id') ? 'data-test-id' : null),
    visible: !!(el.getClientRects().length && getComputedStyle(el).visibility !== 'hidden')
  }));
}"""
        )

        nodes = []
        for i, item in enumerate(js_result):
            node = Node(
                ref=f"e{i}",
                role=item["role"],
                name=item["name"],
                tag=item["tag"],
                placeholder=item["placeholder"],
                testid=item["testid"],
                visible=item["visible"],
                testid_attr=item["testid_attr"],
            )
            nodes.append(node)

        return nodes

    def locator(self, selector: Selector) -> Any:
        """Delegate to build_locator."""
        return build_locator(self._page, selector)

    def act(
        self, selector: Selector, verb: str, value: str | None, timeout_ms: int
    ) -> None:
        """
        Resolve the locator and perform the action.
        Supported verbs: click, fill, wait_visible.
        """
        locator = self.locator(selector)

        if verb == "click":
            locator.first.click(timeout=timeout_ms)
        elif verb == "fill":
            locator.first.fill(value or "", timeout=timeout_ms)
        elif verb == "wait_visible":
            locator.first.wait_for(state="visible", timeout=timeout_ms)
        else:
            raise SelectorConfigError(f"Unknown verb '{verb}'")

    def wait_attached(self, selector: Selector, timeout_ms: int) -> None:
        """Wait for the element to be attached to the DOM."""
        locator = self.locator(selector)
        locator.first.wait_for(state="attached", timeout=timeout_ms)

    def screenshot(self, path: str | Path) -> None:
        """Take a screenshot and save it to the given path."""
        self._page.screenshot(path=str(path))
