#!/usr/bin/env python3
"""
Study Control Panel -- Web UI Server

Serves:
  GET /  ->  index.html (the control panel)

Usage:
  python3 ui_server.py           # default port 8090
  python3 ui_server.py 9000      # custom port

Same bare-stdlib http.server pattern as
block-painting-helper-study/bph_userinterface/bph_ui_server.py -- no extra
web framework dependency.
"""

import sys
import os
from http.server import HTTPServer, BaseHTTPRequestHandler


PORT = int(sys.argv[1]) if len(sys.argv) > 1 else 8090
STATIC_DIR = os.path.dirname(os.path.abspath(__file__))


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ('/', '/index.html'):
            self._serve_file('index.html')
        else:
            self.send_response(404)
            self.send_header('Content-Type', 'text/plain')
            self.end_headers()
            self.wfile.write(b'404 Not Found')

    def _serve_file(self, name):
        path = os.path.join(STATIC_DIR, name)
        try:
            with open(path, 'rb') as f:
                data = f.read()
            self.send_response(200)
            self.send_header('Content-Type', 'text/html; charset=utf-8')
            self.send_header('Content-Length', str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        except FileNotFoundError:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        print(f'[study_control_panel] {self.address_string()} - {fmt % args}')


if __name__ == '__main__':
    server = HTTPServer(('0.0.0.0', PORT), Handler)
    print(f'Study control panel running at http://localhost:{PORT}/')
    print('Press Ctrl-C to stop.')
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
