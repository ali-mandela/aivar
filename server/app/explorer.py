from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, asdict, field
from typing import Any
from urllib.parse import urljoin, urlparse

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


def dismiss_consent(page: Any, timeout_ms: int = 1500) -> str | None:
    """
    Dismiss a consent/cookie banner by clicking a consent button.

    Modal overlays are a known failure mode for crawlers: a banner sits on top
    of the page and blocks clicks on the actual content. This function dismisses
    such banners so the rest of the exploration can proceed.

    Strategy (in order):
    1. Try buttons/links whose accessible name case-insensitively matches
       one of CONSENT_TEXTS (in order, preferring shorter, more specific matches).
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
                locator = page.get_by_role("button", name=consent_text)
                if locator.first.is_visible(timeout=timeout_ms):
                    # Found a button with this accessible name
                    accessible_name = locator.first.get_attribute("aria-label") or consent_text
                    locator.first.click(timeout=timeout_ms)
                    return accessible_name.strip() if accessible_name else consent_text
            except Exception:
                pass

            # Also try links
            try:
                locator = page.get_by_role("link", name=consent_text)
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
    page = None
    browser_wrapper = None

    pages: list[PageObservation] = []
    errors: list[str] = []
    login_form: FormObservation | None = None
    authenticated = False
    consent_dismissed: str | None = None
    skipped_controls: list[str] = []

    visited_urls = set()
    to_visit = []  # (url, depth) - starts empty, we add the entry URL after processing it

    try:
        # Launch browser
        playwright = sync_playwright().start()
        browser = playwright.chromium.launch(headless=headless)
        page = browser.new_page()
        browser_wrapper = Browser(page)

        # Navigate to entry page
        try:
            page.goto(url, wait_until="domcontentloaded")
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
        visited_urls.add(current_url)

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
                    current_url = page.url
                    snap_nodes = browser_wrapper.snapshot()
                    has_password_field = any(n.tag == "input" and n.placeholder == "Password" for n in snap_nodes)

                    if not has_password_field or current_url != url:
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
                if post_login_url not in visited_urls:
                    post_login_page = _extract_page_data(
                        page, browser_wrapper, post_login_url, 0, url
                    )
                    pages.append(post_login_page)
                    visited_urls.add(post_login_url)
                    seed_page = post_login_page
                    logger.info(
                        "Post-login page observed: %s (%d links)",
                        post_login_url, len(post_login_page.links),
                    )
            except Exception as e:
                logger.warning("Could not observe post-login page: %s", e)

        # Seed the crawl queue from whichever page we actually ended up on.
        for link in seed_page.links:
            if link not in visited_urls:
                if not same_origin_only or _is_same_origin(url, link):
                    to_visit.append((link, 1))

        # Breadth-first crawl
        to_visit_idx = 0
        while to_visit_idx < len(to_visit) and len(pages) < max_pages:
            current_url, depth = to_visit[to_visit_idx]
            to_visit_idx += 1

            if current_url in visited_urls or depth > max_depth:
                continue

            visited_urls.add(current_url)

            # Skip logout links to preserve session
            if _is_logout_link(current_url):
                logger.info(f"Skipping logout link (would destroy session): {current_url}")
                continue

            try:
                page.goto(current_url, wait_until="domcontentloaded")
                current_page_url = page.url

                # Observe the page
                reached_by = url if len(pages) > 1 else None  # Track where this page was reached from
                page_obs = _extract_page_data(page, browser_wrapper, current_page_url, depth, reached_by)
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
                    if link not in visited_urls and len(pages) < max_pages and depth < max_depth:
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
