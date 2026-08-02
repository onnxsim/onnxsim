#!/usr/bin/env python3
"""Garbage-collect old ``.dev`` releases from the TestPyPI project.

Every push to ``master`` uploads the full wheel matrix to TestPyPI as a new
``X.Y.Z.devN`` release (see ``.github/workflows/build-and-test.yml``). These
pile up quickly -- the recent ``0.6.5.dev*`` builds are ~375 MB each -- and
TestPyPI enforces a 10 GB per-project limit. This script prunes the old dev
builds so the project stays under the cap.

By default it deletes every ``.dev`` release *except* the newest few, and never
touches real tagged releases (``0.4.x``, ``0.5.0``, ...).

TestPyPI has no delete API (OIDC / trusted-publishing tokens are upload-only),
so deletion has to go through the authenticated web form. TestPyPI also
requires 2FA, which rules out scripted username/password login. Instead this
script reuses your *browser session cookie*, so you log in (and clear 2FA) in
the browser once and paste the cookie here.

Reauthentication (why deletes were silently failing)
----------------------------------------------------
Deleting a release is a "sensitive" action, so Warehouse (the TestPyPI code)
marks the delete view ``require_reauth=True``. A session cookie only satisfies
that if it was *authenticated recently* (the reauth window is ~30 min). When it
wasn't, the delete POST is answered with a redirect to
``/account/reauthenticate/`` and nothing is deleted -- which used to surface as
the opaque ``still present after delete attempt (HTTP 200)`` error.

This script now handles that automatically: on a reauth redirect it
re-authenticates using your account *password* (single factor -- 2FA is *not*
re-checked at reauth) and retries the delete. Provide the password via
``--password`` or ``TESTPYPI_PASSWORD``. If you don't, the script instead tells
you to copy a freshly logged-in cookie.

Getting the cookie
------------------
1. Log in at https://test.pypi.org/ in your browser.
2. Open DevTools -> Network, click any request to test.pypi.org, and copy the
   whole ``Cookie:`` request header. (Or in the Application/Storage tab, copy
   the ``session_id`` cookie value and pass ``session_id=<value>``.)
3. Provide it via ``--cookie`` or the ``TESTPYPI_COOKIE`` environment variable.
   Quote it -- it contains ``;`` and spaces.

Usage
-----
    # See what would be deleted (no changes made):
    python scripts/gc_testpypi.py --dry-run

    # Actually delete, keeping the 3 newest dev builds, asking for confirmation:
    export TESTPYPI_COOKIE='session_id=...; ...'
    export TESTPYPI_PASSWORD='...'          # lets the script reauthenticate
    python scripts/gc_testpypi.py --keep 3

    # Non-interactive:
    python scripts/gc_testpypi.py --keep 3 --yes \
        --cookie "session_id=..." --password "..."

Only ``requests`` is required (``packaging`` is used if present for correct
version sorting, otherwise a built-in fallback is used).
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time

import requests

BASE = "https://test.pypi.org"
DEFAULT_PROJECT = "onnxsim"

# Warehouse marks its sensitive views (including release deletion) as
# ``require_reauth=True`` and redirects unauthenticated-recently sessions here
# instead of performing the action.
REAUTH_PATH = "/account/reauthenticate/"
LOGIN_PATH = "/account/login/"

# Warehouse renders the CSRF token as a hidden form input on every page.
_CSRF_RE = re.compile(r'name="csrf_token"[^>]*value="([^"]+)"')

# Hidden ``<input>`` fields we need to echo back when submitting the reauth
# form. ``next_route*`` all carry ``InputRequired`` validators, so the POST is
# rejected unless we send them back verbatim.
_REAUTH_FIELDS = ("csrf_token", "next_route", "next_route_matchdict",
                  "next_route_query")
_INPUT_RE = re.compile(r"<input\b[^>]*>", re.IGNORECASE)
_ATTR_RE = re.compile(r'([\w:-]+)\s*=\s*"([^"]*)"')


def _parse_cookie_header(cookie: str) -> dict:
    """Parse a ``Cookie:`` header value into a ``{name: value}`` dict."""
    cookie = cookie.strip()
    if cookie.lower().startswith("cookie:"):
        cookie = cookie.split(":", 1)[1].strip()
    jar = {}
    for part in cookie.split(";"):
        part = part.strip()
        if not part or "=" not in part:
            continue
        name, value = part.split("=", 1)
        jar[name.strip()] = value.strip()
    return jar


def _reauth_form_fields(html: str) -> dict:
    """Pull the hidden inputs the reauth form needs echoed back."""
    fields = {}
    for tag in _INPUT_RE.findall(html):
        attrs = dict(_ATTR_RE.findall(tag))
        name = attrs.get("name")
        if name in _REAUTH_FIELDS and name not in fields:
            fields[name] = attrs.get("value", "")
    return fields


def human(nbytes: int) -> str:
    step = 1000.0
    for unit in ("B", "KB", "MB", "GB"):
        if nbytes < step or unit == "GB":
            return f"{nbytes:.1f} {unit}" if unit != "B" else f"{nbytes} B"
        nbytes /= step
    return f"{nbytes:.1f} GB"


def _version_key(version: str):
    """Sort key for release versions, newest last.

    Prefers ``packaging.version`` for PEP 440 correctness; falls back to a
    tuple of ints extracted from the string so we degrade gracefully without
    the dependency.
    """
    try:
        from packaging.version import Version

        return (0, Version(version))
    except Exception:
        nums = tuple(int(n) for n in re.findall(r"\d+", version))
        return (1, nums, version)


def fetch_releases(project: str) -> dict[str, list[dict]]:
    """Return {version: [file-info, ...]} from the public JSON API."""
    url = f"{BASE}/pypi/{project}/json"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    return resp.json()["releases"]


def select_targets(releases: dict[str, list[dict]], keep: int, prune_tagged: bool):
    """Split releases into (to_delete, kept_dev, tagged) lists of versions."""
    dev = sorted((v for v in releases if ".dev" in v), key=_version_key)
    tagged = sorted((v for v in releases if ".dev" not in v), key=_version_key)

    kept_dev = dev[-keep:] if keep > 0 else []
    to_delete = [v for v in dev if v not in set(kept_dev)]

    if prune_tagged and tagged:
        # Keep only the newest tagged release; delete the rest.
        to_delete += tagged[:-1]

    return to_delete, kept_dev, tagged


class TestPyPISession:
    def __init__(self, cookie: str, project: str, password: str | None = None):
        self.project = project
        self.password = password
        self._reauthed = False
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "onnxsim-gc-testpypi/1.0"
        # Load the cookie into the jar rather than pinning a fixed ``Cookie``
        # header: reauthentication may hand us a rotated ``session_id`` via
        # ``Set-Cookie``, and only the jar picks that up automatically.
        jar = _parse_cookie_header(cookie)
        if not jar:
            raise RuntimeError("Could not parse any cookies from the supplied value.")
        requests.utils.add_dict_to_cookiejar(self.s.cookies, jar)

    def _manage_url(self, version: str) -> str:
        return f"{BASE}/manage/project/{self.project}/release/{version}/"

    def _get_release_page(self, url: str) -> str:
        """Load the release-management page and return its CSRF token."""
        resp = self.s.get(url, timeout=30, allow_redirects=True)
        if LOGIN_PATH in resp.url:
            raise RuntimeError(
                "Redirected to the login page -- the session cookie is missing "
                "or expired. Re-copy it from a logged-in browser."
            )
        resp.raise_for_status()
        m = _CSRF_RE.search(resp.text)
        if not m:
            raise RuntimeError(
                f"Could not find a CSRF token on {url}. The page may not exist "
                "or the cookie is not for a maintainer of this project."
            )
        return m.group(1)

    def _reauthenticate(self, next_path: str) -> None:
        """Re-establish a fresh authentication using the account password.

        Warehouse only re-checks the password at reauth (not 2FA), so this is
        scriptable even for a 2FA-protected account.
        """
        if not self.password:
            raise RuntimeError(
                "Deleting a release requires a recently-authenticated session, "
                "but this one is stale (Warehouse redirected to the reauth "
                "page). Either pass your TestPyPI password via --password / "
                "TESTPYPI_PASSWORD so the script can reauthenticate, or log in "
                "again in the browser and copy a FRESH cookie (the reauth "
                "window is only ~30 minutes)."
            )
        url = f"{BASE}{REAUTH_PATH}"
        page = self.s.get(url, params={"next": next_path}, timeout=30,
                          allow_redirects=True)
        if LOGIN_PATH in page.url:
            raise RuntimeError(
                "Redirected to login while reauthenticating -- the session "
                "cookie has expired. Copy a fresh one from the browser."
            )
        page.raise_for_status()
        fields = _reauth_form_fields(page.text)
        if "csrf_token" not in fields:
            raise RuntimeError(
                "Could not find the reauthentication form -- Warehouse may have "
                "changed its layout."
            )
        fields["password"] = self.password
        resp = self.s.post(
            url,
            data=fields,
            headers={"Referer": page.url, "Origin": BASE},
            timeout=30,
            allow_redirects=False,
        )
        # A successful reauth records the timestamp and 303-redirects onward. A
        # rejected password re-renders the form in place (HTTP 200).
        if not resp.is_redirect:
            if resp.status_code == 200:
                raise RuntimeError(
                    "Reauthentication failed -- TestPyPI rejected the password "
                    "(check --password / TESTPYPI_PASSWORD)."
                )
            resp.raise_for_status()
            raise RuntimeError(
                f"Unexpected reauthentication response (HTTP {resp.status_code})."
            )
        self._reauthed = True

    def _post_delete(self, url: str, version: str) -> requests.Response:
        """POST the delete form once. Returns the (un-followed) response."""
        csrf = self._get_release_page(url)
        return self.s.post(
            url,
            data={"csrf_token": csrf, "confirm_delete_version": version},
            headers={"Referer": url, "Origin": BASE},
            timeout=30,
            allow_redirects=False,
        )

    def delete_release(self, version: str) -> None:
        """Delete a single release via the web form. Raises on failure."""
        url = self._manage_url(version)
        next_path = f"/manage/project/{self.project}/release/{version}/"

        resp = self._post_delete(url, version)
        loc = resp.headers.get("Location", "") if resp.is_redirect else ""

        if REAUTH_PATH in loc:
            # Stale session -- reauthenticate and retry the delete once.
            self._reauthenticate(next_path)
            resp = self._post_delete(url, version)
            loc = resp.headers.get("Location", "") if resp.is_redirect else ""
            if REAUTH_PATH in loc:
                raise RuntimeError(
                    f"Still redirected to reauth after reauthenticating "
                    f"{version} -- reauthentication did not take effect."
                )

        if LOGIN_PATH in loc:
            raise RuntimeError(
                f"Redirected to login while deleting {version} -- the session "
                "cookie has expired."
            )

        if not resp.is_redirect:
            # No redirect means the page was re-rendered in place, typically
            # with a flash error (e.g. a confirmation-value mismatch).
            resp.raise_for_status()
            if "Could not delete release" in resp.text:
                raise RuntimeError(
                    f"Server refused to delete {version} (flash error)."
                )

        # Confirm it's really gone: the release page should now 404.
        check = self.s.get(url, timeout=30, allow_redirects=False)
        if check.status_code == 200 and _CSRF_RE.search(check.text):
            raise RuntimeError(
                f"{version} still present after delete attempt (HTTP 200)."
            )


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--project", default=DEFAULT_PROJECT,
                   help=f"TestPyPI project name (default: {DEFAULT_PROJECT})")
    p.add_argument("--keep", type=int, default=3,
                   help="Number of newest .dev releases to keep (default: 3)")
    p.add_argument("--prune-tagged", action="store_true",
                   help="Also delete all but the newest real tagged release.")
    p.add_argument("--cookie", default=os.environ.get("TESTPYPI_COOKIE"),
                   help="TestPyPI session cookie (or set TESTPYPI_COOKIE).")
    p.add_argument("--password", default=os.environ.get("TESTPYPI_PASSWORD"),
                   help="TestPyPI account password (or set TESTPYPI_PASSWORD). "
                        "Used only to reauthenticate a stale session when the "
                        "delete form demands it; 2FA is not re-checked.")
    p.add_argument("--dry-run", action="store_true",
                   help="Show what would be deleted without deleting anything.")
    p.add_argument("--yes", action="store_true",
                   help="Skip the interactive confirmation prompt.")
    args = p.parse_args(argv)

    releases = fetch_releases(args.project)
    to_delete, kept_dev, tagged = select_targets(
        releases, args.keep, args.prune_tagged
    )

    def size_of(v):
        return sum(f["size"] for f in releases.get(v, []))

    total = sum(size_of(v) for v in releases)
    freed = sum(size_of(v) for v in to_delete)

    print(f"Project: {args.project}   TestPyPI limit: 10 GB")
    print(f"Current: {len(releases)} releases, {human(total)}\n")

    print(f"Keeping {len(tagged)} tagged release(s) and "
          f"{len(kept_dev)} newest dev build(s):")
    for v in tagged[-5:]:
        print(f"  keep  {v:<24} {human(size_of(v))}")
    if len(tagged) > 5:
        print(f"  ...   (+{len(tagged) - 5} more tagged)")
    for v in kept_dev:
        print(f"  keep  {v:<24} {human(size_of(v))}")

    print(f"\nDeleting {len(to_delete)} release(s), freeing {human(freed)} "
          f"-> {human(total - freed)} remaining:")
    for v in sorted(to_delete, key=_version_key):
        print(f"  DROP  {v:<24} {human(size_of(v))}")

    if not to_delete:
        print("\nNothing to delete. Already clean.")
        return 0

    if args.dry_run:
        print("\n[dry-run] No changes made.")
        return 0

    if not args.cookie:
        print("\nERROR: no session cookie. Pass --cookie or set TESTPYPI_COOKIE.",
              file=sys.stderr)
        print("See the module docstring for how to obtain it.", file=sys.stderr)
        return 2

    if not args.yes:
        ans = input(f"\nDelete these {len(to_delete)} releases? This is "
                    f"IRREVERSIBLE. [y/N] ").strip().lower()
        if ans not in ("y", "yes"):
            print("Aborted.")
            return 1

    if not args.password:
        print("\nNote: no --password / TESTPYPI_PASSWORD set. Deletes will fail "
              "if the session is not freshly authenticated, since TestPyPI "
              "requires reauthentication for this action.", file=sys.stderr)

    session = TestPyPISession(args.cookie, args.project, args.password)
    ok, failed = 0, 0
    for v in sorted(to_delete, key=_version_key):
        try:
            session.delete_release(v)
            print(f"  deleted {v}")
            ok += 1
        except Exception as exc:  # noqa: BLE001 - report and continue
            print(f"  FAILED  {v}: {exc}", file=sys.stderr)
            failed += 1
        time.sleep(1)  # be polite to the server

    print(f"\nDone. {ok} deleted, {failed} failed. "
          f"Freed ~{human(freed)} (best case).")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
