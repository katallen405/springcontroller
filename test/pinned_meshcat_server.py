#!/home/katallen/.springcontroller_venv/bin/python3
"""
pinned_meshcat_server.py

Standalone meshcat viewer server with BOTH the zmq_url and the HTTP/static
fileserver port pinned to fixed values, so other processes (e.g. the
study_control_panel_ui's embedded iframe) can rely on a predictable URL.

The `meshcat-server` console script (meshcat.servers.zmqserver.main) does
NOT expose a --port flag at all -- it always auto-picks the HTTP port by
searching upward from DEFAULT_FILESERVER_PORT (7000), only using something
else if that's already taken. This script instead constructs
ZMQWebSocketBridge directly, passing port= explicitly, so the HTTP port is
a hard guarantee rather than "probably 7000."
"""
import argparse

from meshcat.servers.zmqserver import ZMQWebSocketBridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--zmq-url", required=True)
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()

    bridge = ZMQWebSocketBridge(zmq_url=args.zmq_url, port=args.port)
    print(f"zmq_url={bridge.zmq_url}")
    print(f"web_url={bridge.web_url}")
    try:
        bridge.run()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
