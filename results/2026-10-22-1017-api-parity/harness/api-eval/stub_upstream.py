import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer
class H(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/health":
            body = b""
        elif self.path == "/v1/models":
            body = json.dumps({"object": "list", "data": [{"id": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "object": "model", "max_model_len": 500000}]}).encode()
        else:
            self.send_response(404); self.end_headers(); return
        self.send_response(200); self.send_header("content-type", "application/json"); self.send_header("content-length", str(len(body))); self.end_headers(); self.wfile.write(body)
    def log_message(self, *a): pass
HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
