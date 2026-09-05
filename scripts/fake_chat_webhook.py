#!/usr/bin/env python
"""A local stand-in for a Google Chat incoming webhook.

Why this exists: **Google Chat incoming webhooks are a Google Workspace
feature.** In a space owned by a personal @gmail.com account there is no
"Apps & integrations -> Webhooks" menu, so there is no webhook URL to point
CHAT_WEBHOOK_URL at. Rather than leave the notify stage untested on a personal
account, point it here instead:

    python scripts/fake_chat_webhook.py                 # terminal 1
    CHAT_WEBHOOK_URL=http://localhost:8099/webhook ...  # terminal 2

It accepts the POST, pretty-prints the payload, renders the card as readable
text, and saves the raw JSON. That exercises the real `chat_notify` code path --
card construction, HTML escaping, the collapsed section, the retry wrapper --
everything except Google's own rendering.

Stdlib only: no dependency on the project's venv.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import sys
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

_TAG = re.compile(r"<[^>]+>")


def _plain(text: str) -> str:
    """Chat renders a small HTML subset in decoratedText. Strip the tags and
    unescape the entities so the terminal shows what a reader would see.

    `html.unescape` rather than a few `.replace()` calls: chat_notify escapes
    with `html.escape`, which also produces `&quot;` and `&#x27;` -- and Sentry
    titles are full of quotes."""
    return html.unescape(_TAG.sub("", text or ""))


def render_card(payload: dict) -> str:
    """Flatten a cardsV2 payload into something readable in a terminal."""
    lines: list[str] = []

    if "text" in payload and "cardsV2" not in payload:
        # The fallback digest and the error notice are plain text messages.
        return _plain(payload["text"])

    for entry in payload.get("cardsV2") or []:
        card = entry.get("card") or {}
        header = card.get("header") or {}
        if header:
            lines.append(f"┌─ {header.get('title', '')} — {header.get('subtitle', '')}")
        for section in card.get("sections") or []:
            if section.get("header"):
                marker = "▸" if section.get("collapsible") else "•"
                lines.append(f"│ {marker} [{_plain(section['header'])}]")
            for widget in section.get("widgets") or []:
                lines.extend(_render_widget(widget))
        lines.append("└" + "─" * 60)
    return "\n".join(lines)


def _render_widget(widget: dict) -> list[str]:
    out: list[str] = []
    if "decoratedText" in widget:
        dt = widget["decoratedText"]
        if dt.get("topLabel"):
            out.append(f"│   {_plain(dt['topLabel'])}")
        out.append(f"│   {_plain(dt.get('text', ''))}")
        if dt.get("bottomLabel"):
            out.append(f"│     {_plain(dt['bottomLabel'])}")
        if dt.get("button"):
            url = ((dt["button"].get("onClick") or {}).get("openLink") or {}).get("url", "")
            out.append(f"│     [{dt['button'].get('text')}] {url}")
        out.append("│")
    for button in (widget.get("buttonList") or {}).get("buttons") or []:
        url = ((button.get("onClick") or {}).get("openLink") or {}).get("url", "")
        out.append(f"│   [{button.get('text')}] {url}")
    return out


class Handler(BaseHTTPRequestHandler):
    out_dir = Path("artifacts/chat")
    fail_times = 0
    _seen = 0

    def do_POST(self) -> None:  # noqa: N802  (stdlib naming)
        raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))

        # Exercise chat_notify's retry wrapper on demand: the first N posts get
        # a 503, which is a retryable status for httpx.HTTPStatusError.
        Handler._seen += 1
        if Handler._seen <= Handler.fail_times:
            print(f"\n>>> returning 503 (failure {Handler._seen}/{Handler.fail_times}) to exercise the retry\n")
            self.send_response(503)
            self.end_headers()
            return

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            print("\n!!! body was not JSON:\n", raw[:2000])
            self.send_response(400)
            self.end_headers()
            return

        stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        self.out_dir.mkdir(parents=True, exist_ok=True)
        path = self.out_dir / f"chat-{stamp}.json"
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        print("\n" + "=" * 64)
        print(f"POST {self.path}   ({len(raw)} bytes)   saved -> {path}")
        print("=" * 64)
        print(render_card(payload))
        print()

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b"{}")

    def log_message(self, *args) -> None:  # silence the default access log
        pass


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--port", type=int, default=8099)
    parser.add_argument(
        "--fail-times",
        type=int,
        default=0,
        help="return 503 for the first N posts, to exercise chat_notify's retry",
    )
    parser.add_argument("--out-dir", default="artifacts/chat")
    args = parser.parse_args()

    Handler.fail_times = args.fail_times
    Handler.out_dir = Path(args.out_dir)

    # Line-buffer stdout: without this, running this server in the background
    # or piping it to a file swallows every card until the process exits, which
    # defeats the point of watching it live.
    sys.stdout.reconfigure(line_buffering=True)

    print(f"fake Google Chat webhook listening on http://localhost:{args.port}/webhook")
    print(f"payloads saved to {Handler.out_dir}/")
    print("point the pipeline at it with:")
    print(f"  CHAT_WEBHOOK_URL=http://localhost:{args.port}/webhook")
    print("\nwaiting for posts (ctrl-c to stop)...")
    try:
        HTTPServer(("localhost", args.port), Handler).serve_forever()
    except KeyboardInterrupt:
        print("\nstopped.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
