from __future__ import annotations

import os
import socket

import uvicorn

from server import app


def _port_free(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        try:
            s.bind(("127.0.0.1", port))
        except OSError:
            return False
        return True


def main() -> None:
    host = "127.0.0.1"
    port = int(os.environ.get("PORT", "8000"))
    while port < 8100 and not _port_free(port):
        port += 1

    print(f"Serving on http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
