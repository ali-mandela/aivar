from __future__ import annotations

import hashlib
import json
import logging
import re
import time
from dataclasses import dataclass, asdict, field, replace
from typing import Any
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

from playwright.sync_api import sync_playwright

from app.browser import Browser, build_locator
from app.config import Guardrails, DEFAULTS
from app.models.models import Selector
from app.resolve import selector_for

logger = logging.getLogger("aivar")

# Consent/cookie banner dismissal
CONSENT_TEXTS = ("accept all", "accept cookies", "allow all", "i agree", "got it", "accept", "agree", "ok")

# Destructive action keywords — used to guard against real submissions
SUBMIT_WORDS = ("submit", "send", "buy", "pay", "order", "checkout", "delete", "remove", "confirm", "subscribe", "register", "sign up")


@dataclass(frozen=True)
class FormField:
    name: str          # best available: label, placeholder, aria-label, name attr, or id
    field_type: str    # text | password | email | checkbox | radio | select | textarea | submit
    required: bool
    selector: Selector # how to reach it, via resolve.selector_for-style preference


@dataclass(frozen=True)
class FormObservation:
    name: str                  # heuristic name, e.g. "login form", "search form", or "form 1"
    fields: list[FormField]
    submit: Selector | None
    is_login: bool             # True when it contains exactly one password field


@dataclass(frozen=True)
class PageObservation:
    url: str
    title: str
    depth: int
    node_count: int
    forms: list[FormObservation]
    links: list[str]           # same-origin absolute URLs found on the page
    headings: list[str]        # visible h1-h3 text, in document order, max 10
    controls: list[str]        # short descriptions of interactive controls, e.g. "button: Add to cart", max 25
    reached_by: str | None     # None for the entry page, else the URL it was reached from
    overlays: list[str] = field(default_factory=list)
    """Controls that revealed a new screen without changing the URL.

    Dialogs, drawers and tab panels have no address of their own, so they can
    never appear in `links`. Naming them here is the only way the Planner learns
    they exist -- and they are frequently the most test-worthy part of an app.
    """


@dataclass
class ExplorationReport:
    entry_url: str
    authenticated: bool
    login_form: FormObservation | None
    pages: list[PageObservation]
    errors: list[str]          # pages that failed to load, with the reason
    duration_ms: float
    consent_dismissed: str | None = None  # What was clicked to dismiss consent banner
    safe_mode: bool = False               # Whether this run was in safe_mode
    skipped_controls: list[str] = field(default_factory=list)  # Controls not clicked due to safe_mode

    def to_dict(self) -> dict:
        """Convert report to a dictionary for JSON serialization."""
        return {
            "entry_url": self.entry_url,
            "authenticated": self.authenticated,
            "login_form": asdict(self.login_form) if self.login_form else None,
            "pages": [asdict(page) for page in self.pages],
            "errors": self.errors,
            "duration_ms": self.duration_ms,
            "consent_dismissed": self.consent_dismissed,
            "safe_mode": self.safe_mode,
            "skipped_controls": self.skipped_controls,
        }

    @property
    def page_count(self) -> int:
        """Return the number of pages discovered."""
        return len(self.pages)

    def summarize(self, max_pages: int = 25, max_chars: int = 6000) -> str:
        """
        Produce a compact plain-text digest for LLM consumption.

        Format: one block per page with url, title, headings, forms (name + field names)
        and controls. Truncate to max_chars on a page boundary.
        """
        lines = []
        char_count = 0

        # Entry info
        entry_line = f"Entry: {self.entry_url}\n"
        lines.append(entry_line)
        char_count += len(entry_line)

        if self.authenticated:
            auth_line = "Authenticated: yes\n"
            lines.append(auth_line)
            char_count += len(auth_line)

        # Page blocks
        for page_idx, page in enumerate(self.pages[:max_pages]):
            # Start a new page block
            page_block = []

            # URL and title
            page_block.append(f"\nPage {page_idx + 1}: {page.url}")
            if page.title:
                page_block.append(f"  Title: {page.title}")

            # Headings
            if page.headings:
                page_block.append(f"  Headings: {', '.join(page.headings)}")

            # Forms
            if page.forms:
                for form in page.forms:
                    form_desc = f"  Form '{form.name}' ({'login' if form.is_login else 'regular'})"
                    field_names = [f.name for f in form.fields]
                    form_desc += f": {', '.join(field_names)}"
                    page_block.append(form_desc)

            # Controls
            if page.controls:
                controls_str = ", ".join(page.controls[:10])  # Limit to 10 for brevity
                page_block.append(f"  Controls: {controls_str}")

            # Screens with no address of their own. Told to the Planner
            # explicitly, because nothing in the URL list implies they exist.
            if page.overlays:
                overlay_str = ", ".join(page.overlays[:8])
                page_block.append(
                    f"  Opens in-page (dialog/panel, no URL change): {overlay_str}"
                )

            # Check if adding this block would exceed max_chars
            block_text = "\n".join(page_block)
            if char_count + len(block_text) > max_chars:
                break

            lines.append(block_text)
            char_count += len(block_text)

        return "\n".join(lines)


def _get_field_name(element: Any) -> str:
    """Extract the best available name for an input field."""
    # Try aria-label
    aria_label = element.get_attribute("aria-label")
    if aria_label and aria_label.strip():
        return aria_label.strip()

    # Try associated label
    element_id = element.get_attribute("id")
    if element_id:
        try:
            label = element.page.query_selector(f'label[for="{element_id}"]')
            if label:
                text = label.text_content()
                if text and text.strip():
                    return text.strip()
        except Exception:
            pass

    # Try placeholder
    placeholder = element.get_attribute("placeholder")
    if placeholder and placeholder.strip():
        return placeholder.strip()

    # Try name attribute
    name_attr = element.get_attribute("name")
    if name_attr and name_attr.strip():
        return name_attr.strip()

    # Try id
    if element_id and element_id.strip():
        return element_id.strip()

    # Fallback: empty string
    return ""


def _get_field_type(element: Any) -> str:
    """Get the field type from an input or textarea element."""
    try:
        tag_name = element.evaluate("el => el.tagName.toLowerCase()")
    except Exception:
        return "text"

    if tag_name == "textarea":
        return "textarea"

    if tag_name == "select":
        return "select"

    if tag_name == "input":
        try:
            input_type = element.get_attribute("type") or "text"
        except Exception:
            input_type = "text"
        input_type = input_type.lower()

        # Map HTML5 types to our simplified set
        if input_type in ("text", "password", "email", "checkbox", "radio", "submit", "button", "reset"):
            return input_type

        # Default unknown input types to text
        return "text"

    return "text"


def _is_required(element: Any) -> bool:
    """Check if a field is marked as required."""
    required_attr = element.get_attribute("required")
    return required_attr is not None


# Test-id attributes in the wild, in preference order. Playwright's own
# get_by_test_id() is hard-wired to a SINGLE configured attribute (data-testid
# by default), so any other convention must be expressed as CSS or the selector
# silently never matches. Saucedemo uses data-test, and getting this wrong here
# meant the explorer compiled selectors that could not resolve and login failed
# on the first real site we pointed it at.
TESTID_ATTRS = ("data-testid", "data-test", "data-test-id")


def _build_selector_for_element(element: Any) -> Selector:
    """Build a usable selector for an element.

    Mirrors resolve.selector_for's stability preference: test id first, then
    accessible name, then placeholder, then structural attributes.
    """
    try:
        # Try testid first, remembering WHICH attribute carried it.
        for attr in TESTID_ATTRS:
            value = element.get_attribute(attr)
            if value and value.strip():
                value = value.strip()
                if attr == "data-testid":
                    return Selector("testid", value)
                return Selector("css", f'[{attr}="{value}"]')

        # Try aria-label
        aria_label = element.get_attribute("aria-label")
        if aria_label and aria_label.strip():
            return Selector("text", aria_label.strip())

        # Try placeholder
        placeholder = element.get_attribute("placeholder")
        if placeholder and placeholder.strip():
            return Selector("placeholder", placeholder.strip())

        # Try id
        element_id = element.get_attribute("id")
        if element_id and element_id.strip():
            return Selector("css", f"#{element_id}")

        # Try name attribute
        name_attr = element.get_attribute("name")
        if name_attr and name_attr.strip():
            return Selector("css", f'[name="{name_attr}"]')

        # Fallback: use CSS selector with tag and type
        tag = element.evaluate("el => el.tagName.toLowerCase()")
        input_type = element.get_attribute("type") or "text"
        return Selector("css", f"{tag}[type='{input_type}']")
    except Exception:
        # Last resort fallback
        return Selector("css", "input")


def _extract_forms(page: Any, browser_wrapper: Browser) -> list[FormObservation]:
    """Extract all forms from the page."""
    forms = []

    # Try to find <form> elements
    form_elements = page.query_selector_all("form")

    if not form_elements:
        # No <form> elements; synthesize a single pseudo-form from all inputs
        # Many login pages (including saucedemo) have no form element
        inputs = page.query_selector_all("input, textarea, select")
        if inputs:
            form_elements = [None]  # Marker for pseudo-form

    for idx, form_element in enumerate(form_elements, start=1):
        if form_element is None:
            # Pseudo-form: all inputs on the page
            field_elements = page.query_selector_all("input, textarea, select")
            form_name = "form 1" if len(form_elements) == 1 else f"form {idx}"
        else:
            # Real form
            field_elements = form_element.query_selector_all("input, textarea, select")
            form_name = form_element.get_attribute("name") or f"form {idx}"

        if not field_elements:
            continue

        # Extract fields
        fields = []
        password_count = 0
        submit_selector = None
        submit_element = None

        for field_element in field_elements:
            field_type = _get_field_type(field_element)

            # Skip submit buttons for now; we'll handle them separately
            if field_type in ("submit", "button", "reset"):
                if field_type == "submit" and submit_selector is None:
                    submit_selector = _build_selector_for_element(field_element)
                    submit_element = field_element
                continue

            field_name = _get_field_name(field_element)
            if not field_name:
                field_name = f"field_{len(fields)}"

            if field_type == "password":
                password_count += 1

            required = _is_required(field_element)

            # Build selector for this field
            field_selector = _build_selector_for_element(field_element)

            fields.append(FormField(
                name=field_name,
                field_type=field_type,
                required=required,
                selector=field_selector,
            ))

        # Detect if this is a login form (exactly one password field)
        is_login = (password_count == 1)

        # Try to find submit button if not found yet
        if submit_selector is None and form_element is not None:
            # Look for submit button within the form
            submit_elem = form_element.query_selector("button[type=submit], button:not([type]), input[type=submit]")
            if submit_elem:
                submit_selector = _build_selector_for_element(submit_elem)
                submit_element = submit_elem
        elif submit_selector is None:
            # For pseudo-form (no form_element), look for submit in all inputs
            submit_elem = page.query_selector("button[type=submit], button:not([type]), input[type=submit]")
            if submit_elem:
                submit_selector = _build_selector_for_element(submit_elem)
                submit_element = submit_elem

        if fields:
            # Derive a meaningful form name
            form_name = _derive_form_name(
                form_name, is_login, fields, submit_element, page
            )

            forms.append(FormObservation(
                name=form_name,
                fields=fields,
                submit=submit_selector,
                is_login=is_login,
            ))

    return forms


def _derive_form_name(
    fallback_name: str,
    is_login: bool,
    fields: list[FormField],
    submit_element: Any | None,
    page: Any,
) -> str:
    """Derive a meaningful name for a form based on its content.

    Strategy:
    1. If exactly one password field (login form), use "login form" (or "login form on <path>" if path is not "/")
    2. Else if submit button has accessible name, use "<submit name> form" lowercased
    3. Else if has fields, use "<first field name> form"
    4. Else fall back to fallback_name (e.g. "form 1")
    """
    # Strategy 1: Login form
    if is_login:
        try:
            page_url = page.url
            from urllib.parse import urlparse
            parsed = urlparse(page_url)
            path = parsed.path
            if path and path != "/":
                return f"login form on {path}"
        except Exception:
            pass
        return "login form"

    # Strategy 2: Submit button name (try aria-label, value, or text content)
    if submit_element is not None:
        try:
            # Try aria-label
            aria_label = submit_element.get_attribute("aria-label")
            if aria_label and aria_label.strip():
                return f"{aria_label.strip()} form".lower()

            # Try value attribute (for input[type=submit])
            value = submit_element.get_attribute("value")
            if value and value.strip():
                return f"{value.strip()} form".lower()

            # Try text content (for button elements)
            text = submit_element.text_content()
            if text and text.strip():
                return f"{text.strip()} form".lower()
        except Exception:
            pass

    # Strategy 3: First field name
    if fields:
        first_field_name = fields[0].name
        if first_field_name and not first_field_name.startswith("field_"):
            return f"{first_field_name} form".lower()

    # Strategy 4: Fallback
    return fallback_name


def _extract_page_data(page: Any, browser_wrapper: Browser, url: str, depth: int, reached_by: str | None) -> PageObservation:
    """Extract all observable data from a page."""
    try:
        title = page.title()
    except Exception:
        title = ""

    # Count nodes
    try:
        node_count = len(browser_wrapper.snapshot())
    except Exception:
        node_count = 0

    # Extract headings
    headings = []
    try:
        heading_elements = page.query_selector_all("h1, h2, h3")
        for elem in heading_elements:
            try:
                # Check if visible
                is_visible = elem.is_visible()
                if is_visible:
                    text = elem.text_content().strip()
                    if text and len(headings) < 10:
                        headings.append(text)
            except Exception:
                pass
    except Exception:
        pass

    # Extract links
    links = []
    try:
        link_elements = page.query_selector_all("a[href]")
        for elem in link_elements:
            try:
                href = elem.get_attribute("href")
                if href:
                    # Resolve relative URLs
                    try:
                        absolute_url = urljoin(url, href)
                        # Only same-origin
                        if _is_same_origin(url, absolute_url):
                            # Normalize: strip fragments
                            if "#" in absolute_url:
                                absolute_url = absolute_url.split("#")[0]
                            if absolute_url not in links and not absolute_url.endswith(".pdf"):
                                links.append(absolute_url)
                    except Exception:
                        pass
            except Exception:
                pass
    except Exception:
        pass

    # Extract forms
    forms = _extract_forms(page, browser_wrapper)

    # Extract controls
    controls = []
    try:
        snap_nodes = browser_wrapper.snapshot()
        for node in snap_nodes:
            if node.role and node.name and node.visible:
                control_desc = f"{node.role}: {node.name}"
                if len(controls) < 25:
                    controls.append(control_desc)
    except Exception:
        pass

    return PageObservation(
        url=url,
        title=title,
        depth=depth,
        node_count=node_count,
        forms=forms,
        links=links,
        headings=headings,
        controls=controls,
        reached_by=reached_by,
    )


def _is_same_origin(url1: str, url2: str) -> bool:
    """Check if two URLs are same-origin."""
    try:
        parsed1 = urlparse(url1)
        parsed2 = urlparse(url2)
        return (parsed1.scheme == parsed2.scheme and
                parsed1.netloc == parsed2.netloc)
    except Exception:
        return False


def _is_logout_link(url: str) -> bool:
    """Check if a URL looks like a logout/signout link."""
    url_lower = url.lower()
    return "logout" in url_lower or "signout" in url_lower or "sign-out" in url_lower


# ---------------------------------------------------------------------------
# Seeing the page the way a person does
#
# A human does not read hrefs, does not treat ?sort=asc as a different screen,
# and does not decide a page is ready the instant the DOM parses. The helpers
# below encode those three habits.
# ---------------------------------------------------------------------------

# Path segments that are record ids rather than distinct screens: /order/1042
# and /order/1043 are one screen a human would test once.
_ID_SEGMENT_RE = re.compile(
    r"^(\d+|[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}|[0-9a-f]{24,})$",
    re.IGNORECASE,
)

# Controls that can reveal a screen. "link" is included because an anchor
# written as href="#" or href="javascript:void(0)" navigates via JavaScript and
# is invisible to href harvesting -- that is the ordinary shape of a
# single-page app. Anchors carrying a real href are filtered out at click time
# (see _href_is_dead): they are already collected, and clicking a footer full of
# social links only opens external tabs.
NAVIGATIONAL_ROLES = ("button", "tab", "menuitem", "link")

# Never click these while exploring: they end the session or leave the app.
_AVOID_CLICK_WORDS = (
    "logout", "log out", "sign out", "signout", "delete", "remove",
    "cancel account", "close account", "deactivate", "download",
)


def _normalize_url(url: str) -> str:
    """A stable key for 'have I already been here'.

    Drops the fragment, sorts query parameters and removes a trailing slash, so
    /products, /products/ and /products?b=2&a=1 collapse to one entry instead of
    spending three of the crawl's page budget.
    """
    try:
        parts = urlparse(url)
        query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
        path = parts.path.rstrip("/") or "/"
        return urlunparse((parts.scheme, parts.netloc, path, "", query, ""))
    except Exception:
        return url


def _screen_signature(url: str) -> str:
    """A key for 'is this the same *kind* of screen'.

    Record ids in the path are replaced with a placeholder and query values are
    dropped, so twenty product pages read as one screen. Without this the page
    budget is spent enumerating rows of a table rather than finding new screens.
    """
    try:
        parts = urlparse(url)
        segments = [
            "{id}" if _ID_SEGMENT_RE.match(seg) else seg
            for seg in parts.path.strip("/").split("/")
            if seg
        ]
        keys = ",".join(sorted(k for k, _ in parse_qsl(parts.query)))
        return f"{parts.netloc}/{'/'.join(segments)}" + (f"?{keys}" if keys else "")
    except Exception:
        return url


def _settle(page: Any, timeout_ms: int = 3000, idle_ms: int = 1200) -> None:
    """Wait until the page looks ready to a person, not merely parsed.

    domcontentloaded fires before the framework has rendered anything, so on a
    React or Vue app a snapshot taken then sees an empty shell. Waiting for the
    network to go quiet is what a human waiting for the spinner to stop is
    actually doing. Both waits are best-effort: a page that streams forever
    still gets explored rather than failing the run.

    `idle_ms` is deliberately much shorter than `timeout_ms`. Apps with polling
    or analytics beacons never reach networkidle at all, and waiting the full
    timeout on each of them turned a six-page crawl into a minute of dead air.
    """
    try:
        page.wait_for_load_state("domcontentloaded", timeout=timeout_ms)
    except Exception:
        pass
    try:
        page.wait_for_load_state("networkidle", timeout=idle_ms)
    except Exception:
        # Long-polling and analytics beacons mean networkidle never arrives on
        # some apps. The DOM wait above already happened; carry on.
        pass


def _screen_digest(browser_wrapper: Browser) -> str:
    """A fingerprint of what is currently on screen.

    Used to tell 'the click opened something new' from 'the click did nothing',
    which is how a person knows a modal appeared even though the URL did not
    change.
    """
    try:
        nodes = browser_wrapper.snapshot()
        parts = sorted(f"{n.role}:{n.name}" for n in nodes if n.visible and n.name)
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()[:16]
    except Exception:
        return ""


def _looks_authenticated(page: Any, browser_wrapper: Browser, before_url: str) -> bool:
    """Decide whether a login attempt worked.

    Checked the way a person checks: the password box is gone, or the address
    changed, or something that only appears when signed in is now on screen.

    The previous implementation compared a placeholder to the exact string
    "Password", so any app writing "password", "Enter your password" or using a
    label instead reported a successful login as failed.
    """
    try:
        still_asking = any(
            el.is_visible() for el in page.query_selector_all("input[type='password']")
        )
    except Exception:
        still_asking = False

    if not still_asking:
        return True

    try:
        if page.url != before_url:
            return True
    except Exception:
        pass

    # A sign-out affordance is the clearest signal a human uses.
    try:
        for node in browser_wrapper.snapshot():
            name = (node.name or "").lower()
            if node.visible and any(w in name for w in ("logout", "log out", "sign out")):
                return True
    except Exception:
        pass

    return False


def _should_avoid_clicking(name: str, safe_mode: bool) -> bool:
    """Whether exploring should leave this control alone."""
    lowered = (name or "").strip().lower()
    if not lowered:
        return True
    if any(word in lowered for word in _AVOID_CLICK_WORDS):
        return True
    if is_destructive(lowered):
        return True
    if safe_mode and any(word in lowered for word in SUBMIT_WORDS):
        return True
    return False


def _href_is_dead(locator: Any) -> bool:
    """Whether an anchor goes nowhere on its own.

    href="#", href="javascript:void(0)" and a missing href all mean the same
    thing: the destination lives in a click handler, not in the markup. Those
    are exactly the anchors href harvesting cannot follow, and on a single-page
    app they are most of them.
    """
    try:
        href = (locator.get_attribute("href") or "").strip()
    except Exception:
        return False
    return href in ("", "#") or href.lower().startswith("javascript:")


def _discover_by_clicking(
    page: Any,
    browser_wrapper: Browser,
    current_url: str,
    *,
    entry_url: str,
    same_origin_only: bool,
    safe_mode: bool,
    max_clicks: int = 8,
) -> tuple[list[str], list[str]]:
    """Find screens that no anchor tag points at.

    Harvesting `a[href]` finds only what a server-rendered site links to. A
    single-page app routes through onClick handlers, and its most test-worthy
    screens -- carts, dialogs, wizards -- often have no URL at all. A person
    finds those by clicking things that look clickable and noticing the screen
    changed; this does the same.

    Returns (urls, overlays): real URLs reached by clicking, and the names of
    controls that changed the screen without navigating (dialogs, drawers,
    tabs). Overlays are reported so the Planner knows they exist even though
    they cannot be reached by navigation.

    The page is returned to `current_url` after every click, because a click
    that navigates would otherwise silently move the crawl somewhere else.
    """
    found_urls: list[str] = []
    overlays: list[str] = []

    try:
        candidates = [
            n
            for n in browser_wrapper.snapshot()
            if n.visible
            and n.role in NAVIGATIONAL_ROLES
            and n.name
            and not _should_avoid_clicking(n.name, safe_mode)
        ]
    except Exception:
        return ([], [])

    # Deduplicate by name: two buttons reading "Add to cart" are one behaviour.
    seen_names: set[str] = set()
    unique: list[Any] = []
    for node in candidates:
        key = node.name.strip().lower()
        if key not in seen_names:
            seen_names.add(key)
            unique.append(node)

    for node in unique[:max_clicks]:
        # Measured immediately before this click, not once at the top: an
        # earlier click can leave the page permanently changed ("Add to cart"
        # becomes "Remove"), and comparing against a stale baseline then reports
        # every later control as having opened something.
        try:
            before = _screen_digest(browser_wrapper)
        except Exception:
            before = ""

        try:
            locator = page.get_by_role(node.role, name=node.name).first
            if not locator.is_visible(timeout=500):
                continue
            # An anchor with a real destination was already recorded from its
            # href. Only click the ones that navigate by script.
            if node.role == "link" and not _href_is_dead(locator):
                continue
            locator.click(timeout=2500)
            _settle(page, timeout_ms=2000, idle_ms=600)
        except Exception:
            # A control that will not click is not an error: it may be disabled,
            # covered, or have moved. Try the next one.
            continue

        navigated = False
        try:
            after_url = page.url
            if _normalize_url(after_url) != _normalize_url(current_url):
                navigated = True
                if not same_origin_only or _is_same_origin(entry_url, after_url):
                    if after_url not in found_urls and not _is_logout_link(after_url):
                        found_urls.append(after_url)
            elif _screen_digest(browser_wrapper) != before:
                # Same address, different screen: a dialog or a tab panel.
                overlays.append(node.name)
                navigated = True  # the screen changed; put it back
        except Exception:
            pass

        # Only pay for a reload when the click actually moved us. Most clicks do
        # nothing at all, and re-navigating after each one was the single
        # largest cost in the crawl.
        if navigated:
            try:
                page.goto(current_url, wait_until="domcontentloaded")
                _settle(page, timeout_ms=2000, idle_ms=600)
            except Exception:
                break

    return (found_urls, overlays)


def dismiss_consent(page: Any, timeout_ms: int = 1500) -> str | None:
    """
    Dismiss a consent/cookie banner by clicking a consent button.

    Modal overlays are a known failure mode for crawlers: a banner sits on top
    of the page and blocks clicks on the actual content. This function dismisses
    such banners so the rest of the exploration can proceed.

    Strategy (in order):
    1. Try buttons/links whose accessible name case-insensitively matches
       one of CONSENT_TEXTS (in order, preferring shorter, more specific matches).
       Matched with exact=True: Playwright's `name=` is a substring match by
       default, so the "ok" entry matched "Br-ok-en Images" and every other
       name containing those two letters. On a link, clicking it navigated away
       before exploration had observed a single page.
    2. Fall back to common ids/classes: #onetrust-accept-btn-handler, .cc-allow,
       [aria-label*="accept" i], [id*="cookie" i] button

    Args:
        page: Playwright page object
        timeout_ms: Timeout for finding and clicking the element

    Returns:
        The accessible name of the clicked button, or None if no banner was found.
        Never raises — a missing banner is the normal case.
    """
    try:
        # Strategy 1: Match accessible names from CONSENT_TEXTS (in order, prefer specific)
        for consent_text in CONSENT_TEXTS:
            try:
                # Use get_by_role to find visible buttons or links with matching accessible name
                # (case-insensitive match)
                locator = page.get_by_role("button", name=consent_text, exact=True)
                if locator.first.is_visible(timeout=timeout_ms):
                    # Found a button with this accessible name
                    accessible_name = locator.first.get_attribute("aria-label") or consent_text
                    locator.first.click(timeout=timeout_ms)
                    return accessible_name.strip() if accessible_name else consent_text
            except Exception:
                pass

            # Also try links
            try:
                locator = page.get_by_role("link", name=consent_text, exact=True)
                if locator.first.is_visible(timeout=timeout_ms):
                    accessible_name = locator.first.get_attribute("aria-label") or consent_text
                    locator.first.click(timeout=timeout_ms)
                    return accessible_name.strip() if accessible_name else consent_text
            except Exception:
                pass

        # Strategy 2: Fall back to common ids/classes
        selectors_to_try = [
            "#onetrust-accept-btn-handler",
            ".cc-allow",
            "[aria-label*='accept' i]",
            "[id*='cookie' i] button",
        ]

        for selector in selectors_to_try:
            try:
                locator = page.locator(selector)
                if locator.first.is_visible(timeout=timeout_ms):
                    # Get the accessible name or text
                    try:
                        accessible_name = locator.first.get_attribute("aria-label")
                    except Exception:
                        accessible_name = None

                    if not accessible_name:
                        try:
                            accessible_name = locator.first.text_content()
                        except Exception:
                            accessible_name = None

                    locator.first.click(timeout=timeout_ms)
                    return (accessible_name.strip() if accessible_name else "consent button")
            except Exception:
                pass

        # No consent banner found
        return None

    except Exception:
        # Never raise on missing banner
        return None


def is_destructive(name: str) -> bool:
    """
    Check if a control name indicates a destructive action.

    Destructive actions include submitting forms, making purchases, deleting data, etc.
    Login/sign-in controls are explicitly NOT considered destructive, as authentication
    is required to explore the application.

    Args:
        name: Accessible name or description of the control

    Returns:
        True if the name matches a destructive action keyword, False otherwise
    """
    name_lower = name.lower()

    # Explicitly exclude login/signin controls (authentication is required for exploration)
    if "log" in name_lower and "in" in name_lower:  # "log in", "login"
        return False
    if "sign" in name_lower and "in" in name_lower:  # "sign in", "signin"
        return False
    if "continue" in name_lower:  # "Continue" (multi-step login)
        return False

    # Check if any destructive keyword is present
    for word in SUBMIT_WORDS:
        if word in name_lower:
            return True

    return False


def explore(
    url: str,
    *,
    username: str | None = None,
    password: str | None = None,
    max_pages: int = 8,
    max_depth: int = 2,
    headless: bool = True,
    guardrails: Guardrails = DEFAULTS,
    same_origin_only: bool = True,
    safe_mode: bool = False,
) -> ExplorationReport:
    """
    A deterministic, auth-aware crawler that discovers what an application contains.

    Args:
        url: Entry URL
        username: Username for login (if a login form is found and both username and password are provided)
        password: Password for login
        max_pages: Maximum number of pages to crawl
        max_depth: Maximum depth of link-hops to crawl
        headless: Whether to run browser in headless mode
        guardrails: Guardrails configuration
        same_origin_only: Whether to only crawl same-origin links
        safe_mode: If True, do not click destructive controls (submit, send, buy, etc.).
                   Use this when crawling third-party sites to avoid sending real emails,
                   making real orders, or deleting data. The explorer will still fill fields
                   and observe forms, but will skip controls that look destructive.
                   Note: --safe-mode

    Returns:
        ExplorationReport with discovered pages, forms, and errors
    """
    start_time = time.perf_counter()
    playwright = None
    browser = None
    context = None
    page = None
    browser_wrapper = None

    pages: list[PageObservation] = []
    errors: list[str] = []
    login_form: FormObservation | None = None
    authenticated = False
    consent_dismissed: str | None = None
    skipped_controls: list[str] = []

    visited_urls: set[str] = set()
    # Screen shapes already seen, so /order/1 does not also spend the page
    # budget on /order/2 ... /order/20. See _screen_signature.
    visited_signatures: set[str] = set()
    to_visit = []  # (url, depth) - starts empty, we add the entry URL after processing it

    try:
        # Launch browser
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        # A context, not a bare page: it is the only place Playwright lets us
        # fix the viewport, and later the seam for tracing and storage_state.
        # Without an explicit viewport this ran at Playwright's 800x600 default
        # while every later stage used 1280x720 -- so exploration saw a
        # collapsed mobile nav and planned against elements the executor would
        # never find.
        context = browser.new_context(
            viewport={"width": 1280, "height": 720},
            locale="en-US",
        )
        page = context.new_page()
        browser_wrapper = Browser(page)

        # Navigate to entry page
        try:
            page.goto(url, wait_until="domcontentloaded")
            _settle(page)
        except Exception as e:
            errors.append(f"Failed to load entry page {url}: {str(e)}")
            return ExplorationReport(
                entry_url=url,
                authenticated=False,
                login_form=None,
                pages=[],
                errors=errors,
                duration_ms=(time.perf_counter() - start_time) * 1000,
                consent_dismissed=None,
                safe_mode=safe_mode,
                skipped_controls=[],
            )

        # Dismiss consent banner if present (enables exploration on sites with cookie banners)
        consent_dismissed = dismiss_consent(page)

        # Observe entry page
        current_url = page.url
        entry_page = _extract_page_data(page, browser_wrapper, current_url, 0, None)
        pages.append(entry_page)
        visited_urls.add(_normalize_url(current_url))
        visited_signatures.add(_screen_signature(current_url))

        # Check for login form on entry page
        if entry_page.forms:
            for form in entry_page.forms:
                if form.is_login:
                    login_form = form
                    break

        # Attempt login if credentials are provided and a login form is present
        if login_form and username and password:
            try:
                # Find the username field (prefer user|email|login|account)
                username_field = None
                password_field = None

                for field in login_form.fields:
                    if field.field_type == "password":
                        password_field = field
                    elif username_field is None and field.field_type in ("text", "email"):
                        # Check if field name matches username patterns
                        field_name_lower = field.name.lower()
                        if any(pat in field_name_lower for pat in ("user", "email", "login", "account")):
                            username_field = field

                # Fallback: use first non-password text field before password field
                if username_field is None and password_field:
                    for field in login_form.fields:
                        if field.field_type in ("text", "email"):
                            username_field = field
                            break

                # Fill and submit
                if username_field and password_field:
                    browser_wrapper.act(username_field.selector, "fill", username, 5000)
                    browser_wrapper.act(password_field.selector, "fill", password, 5000)

                    # Click submit (login submit is never skipped even in safe_mode)
                    if login_form.submit:
                        browser_wrapper.act(login_form.submit, "click", None, 5000)
                    else:
                        # Press Enter in password field
                        password_field.selector
                        page.press(f"{password_field.selector.strategy}={password_field.selector.value}", "Enter")

                    # Wait for navigation or DOM settle
                    try:
                        page.wait_for_load_state("domcontentloaded", timeout=3000)
                    except Exception:
                        pass

                    # Check if authenticated
                    if _looks_authenticated(page, browser_wrapper, url):
                        authenticated = True
                        logger.info(f"Login successful, authenticated={authenticated}")

            except Exception as e:
                logger.warning(f"Login attempt failed: {e}")

        # In safe_mode, record any destructive controls found on the entry page
        if safe_mode:
            try:
                snap_nodes = browser_wrapper.snapshot()
                for node in snap_nodes:
                    if node.role == "button" and node.name:
                        if is_destructive(node.name) and node.name not in skipped_controls:
                            skipped_controls.append(node.name)
                            logger.info(f"Skipping destructive control in safe_mode: {node.name}")
            except Exception:
                pass

        # If logging in moved us somewhere new, observe THAT page too and crawl
        # from it. Without this the queue is seeded only from the pre-login page,
        # and on any app whose login screen has no navigation (saucedemo, most
        # SPAs behind an auth wall) the entire application stays invisible -- the
        # crawl ends with one page and the planner has nothing to work from.
        seed_page = entry_page
        if authenticated:
            try:
                post_login_url = page.url
                if _normalize_url(post_login_url) not in visited_urls:
                    post_login_page = _extract_page_data(
                        page, browser_wrapper, post_login_url, 0, url
                    )
                    pages.append(post_login_page)
                    visited_urls.add(_normalize_url(post_login_url))
                    visited_signatures.add(_screen_signature(post_login_url))
                    seed_page = post_login_page
                    logger.info(
                        "Post-login page observed: %s (%d links)",
                        post_login_url, len(post_login_page.links),
                    )
            except Exception as e:
                logger.warning("Could not observe post-login page: %s", e)

        # The seed page is observed outside the crawl loop, so click discovery
        # has not run on it yet -- and after a login it is the landing page,
        # usually the richest screen in the app. Explore it the same way.
        try:
            seed_clicked, seed_overlays = _discover_by_clicking(
                page,
                browser_wrapper,
                page.url,
                entry_url=url,
                same_origin_only=same_origin_only,
                safe_mode=safe_mode,
            )
            # PageObservation is frozen, so rebuild it and swap it into `pages`.
            for found in seed_clicked:
                if found not in seed_page.links:
                    seed_page.links.append(found)
            updated = replace(seed_page, overlays=seed_overlays)
            for i, observed in enumerate(pages):
                if observed is seed_page:
                    pages[i] = updated
                    break
            seed_page = updated
            if seed_clicked or seed_overlays:
                logger.info(
                    "Seed page: %d screen(s) reached by clicking, %d in-page panel(s)",
                    len(seed_clicked), len(seed_overlays),
                )
        except Exception as e:
            logger.warning("Click discovery on seed page failed: %s", e)

        # Seed the crawl queue from whichever page we actually ended up on.
        for link in seed_page.links:
            if _normalize_url(link) not in visited_urls:
                if not same_origin_only or _is_same_origin(url, link):
                    to_visit.append((link, 1))

        # Breadth-first crawl
        to_visit_idx = 0
        while to_visit_idx < len(to_visit) and len(pages) < max_pages:
            current_url, depth = to_visit[to_visit_idx]
            to_visit_idx += 1

            # Compare normalized: /cart, /cart/ and /cart?ref=nav are one screen,
            # and counting them separately spent the page budget three times over.
            visit_key = _normalize_url(current_url)
            if visit_key in visited_urls or depth > max_depth:
                continue

            # Twenty rows of a table are twenty URLs but one screen. Visit the
            # first, then move on to somewhere genuinely new.
            signature = _screen_signature(current_url)
            if signature in visited_signatures:
                continue

            visited_urls.add(visit_key)
            visited_signatures.add(signature)

            # Skip logout links to preserve session
            if _is_logout_link(current_url):
                logger.info(f"Skipping logout link (would destroy session): {current_url}")
                continue

            try:
                page.goto(current_url, wait_until="domcontentloaded")
                _settle(page)
                current_page_url = page.url

                # Observe the page
                reached_by = url if len(pages) > 1 else None  # Track where this page was reached from
                page_obs = _extract_page_data(page, browser_wrapper, current_page_url, depth, reached_by)

                # Anchors alone miss every SPA route and dialog, so also click
                # the things a person would click and see where they land.
                if depth < max_depth:
                    clicked_urls, overlays = _discover_by_clicking(
                        page,
                        browser_wrapper,
                        current_page_url,
                        entry_url=url,
                        same_origin_only=same_origin_only,
                        safe_mode=safe_mode,
                    )
                    for found in clicked_urls:
                        if found not in page_obs.links:
                            page_obs.links.append(found)
                    # Frozen dataclass: rebuild rather than assign.
                    page_obs = replace(page_obs, overlays=overlays)

                pages.append(page_obs)

                # In safe_mode, record any destructive controls found on this page
                if safe_mode:
                    try:
                        snap_nodes = browser_wrapper.snapshot()
                        for node in snap_nodes:
                            if node.role == "button" and node.name:
                                if is_destructive(node.name) and node.name not in skipped_controls:
                                    skipped_controls.append(node.name)
                                    logger.info(f"Skipping destructive control in safe_mode: {node.name}")
                    except Exception:
                        pass

                # Add links to visit queue
                for link in page_obs.links:
                    if _normalize_url(link) in visited_urls:
                        continue
                    if _screen_signature(link) in visited_signatures:
                        continue
                    if len(pages) < max_pages and depth < max_depth:
                        if not same_origin_only or _is_same_origin(url, link):
                            to_visit.append((link, depth + 1))

            except Exception as e:
                errors.append(f"Failed to load {current_url}: {str(e)}")
                continue

    finally:
        # Always close browser
        if page is not None:
            try:
                page.close()
            except Exception:
                pass
        if context is not None:
            try:
                context.close()
            except Exception:
                pass
        if browser is not None:
            try:
                browser.close()
            except Exception:
                pass
        if playwright is not None:
            try:
                playwright.stop()
            except Exception:
                pass

    duration_ms = (time.perf_counter() - start_time) * 1000

    return ExplorationReport(
        entry_url=url,
        authenticated=authenticated,
        login_form=login_form,
        pages=pages,
        errors=errors,
        duration_ms=duration_ms,
        consent_dismissed=consent_dismissed,
        safe_mode=safe_mode,
        skipped_controls=skipped_controls,
    )
