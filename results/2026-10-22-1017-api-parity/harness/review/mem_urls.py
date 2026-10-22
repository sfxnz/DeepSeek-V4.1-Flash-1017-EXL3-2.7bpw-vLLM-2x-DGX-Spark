#!/usr/bin/env python3
"""Gateway.images() fetches every URL image one by one with a standalone fetch (32 MiB limit per image, no
request total) and keeps all of them before ImageResolver.resolve applies the API's 64 MiB request total and
600-image count. A Chat request with N image URLs to a 32 MiB body: peak RSS and the outcome.
Run inside the serve image (no network; the image server is a child process on 127.0.0.1):

  docker run --rm --network none --memory 2g ... --entrypoint python3 <image> /rv/mem_urls.py N
"""
import http.server
import json
import resource
import subprocess
import sys
import time

sys.path.insert(0, "/gw")
TOK = "/hub/models--sfxnz--DeepSeek-V4.1-Flash-1017-EXL3/snapshots/2.7bpw-mcg-hq/tokenizer.json"
BODY = b"\x89PNG\r\n\x1a\n" + b"\0" * (32 * 2**20 - 64)


class H(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("content-type", "image/png")
        self.send_header("content-length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *a):
        pass


def main():
    if sys.argv[1] == "serve":
        http.server.ThreadingHTTPServer(("127.0.0.1", 18555), H).serve_forever()
        return
    n = int(sys.argv[1])
    srv = subprocess.Popen([sys.executable, __file__, "serve"])
    time.sleep(1)
    import dsv41_gateway as G

    gw = G.Gateway(TOK, "deepseek-ai/DeepSeek-V4.1-Flash-1017")
    body = {"model": "deepseek-ai/DeepSeek-V4.1-Flash-1017", "thinking": {"type": "disabled"},
            "messages": [{"role": "user", "content": [{"type": "text", "text": "Describe."}] + [
                {"type": "image_url", "image_url": {"url": f"http://127.0.0.1:18555/i{i}.png"}} for i in range(n)]}]}
    t0 = time.monotonic()
    try:
        gw.prepare("chat", json.loads(json.dumps(body)))
        outcome = "prepared"
    except G.Fallback as e:
        outcome = f"Fallback({str(e)[:160]!r}) -> vLLM fetches all {n} URLs again"
    except G.ApiError as e:
        outcome = f"ApiError {e.status} {e.body['error']['message'][:160]!r}"
    print(f"N={n} URL images of 32 MiB: {outcome}; {time.monotonic() - t0:.1f} s; "
          f"peak RSS {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024} MiB", flush=True)
    srv.kill()


if __name__ == "__main__":
    main()
