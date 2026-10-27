"""Local-only video encoder output sink; avoids interactive save dialogs."""
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
import json

root = Path(__file__).resolve().parents[1]
dest = root/'results/youtube'
dest.mkdir(exist_ok=True)

class Handler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(root/'web/public'), **kwargs)
    def do_POST(self):
        names = {f'/save/{run}.h264': dest/f'fullscreen-{run}.h264' for run in ('low','high')}
        path = names.get(self.path)
        size = int(self.headers.get('Content-Length',0))
        if not path or not 0 < size < 300_000_000 or path.exists():
            self.send_error(400, 'Invalid output or file already exists'); return
        data = self.rfile.read(size)
        if len(data) != size:
            self.send_error(400, 'Incomplete output'); return
        path.write_bytes(data)
        self.send_response(200); self.end_headers(); self.wfile.write(b'Saved')

ThreadingHTTPServer(('127.0.0.1',8772),Handler).serve_forever()
