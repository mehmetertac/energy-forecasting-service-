"""Minimal HTTP placeholder until the Streamlit dashboard is implemented."""

from __future__ import annotations

from http.server import BaseHTTPRequestHandler, HTTPServer


class _StubHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        body = (
            "Streamlit dashboard placeholder — not built yet.\n"
            "Later: P10–P90 uncertainty bands via POST /forecast.\n"
        ).encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return


def main() -> None:
    server = HTTPServer(("0.0.0.0", 8501), _StubHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
