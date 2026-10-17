"""Minimal Hugging Face Hub access for pinned dataset revisions: file listing, HTTP-range parquet reads
(so only the sampled row groups are downloaded) and whole-file downloads. Standard library + pyarrow."""

from __future__ import annotations

import io
import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

HF = "https://huggingface.co"


def _token():
    tok = os.environ.get("HF_TOKEN")
    if tok:
        return tok.strip()
    p = Path.home() / ".cache" / "huggingface" / "token"
    return p.read_text().strip() if p.exists() else None


def _req(url, headers=None, timeout=120, tries=6):
    h = dict(headers or {})
    tok = _token()
    if tok and url.startswith(HF):
        h["Authorization"] = f"Bearer {tok}"
    last = None
    for i in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=h), timeout=timeout)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (401, 403, 404):
                raise
            last = e
            time.sleep(min(60, 2 ** i))
    raise last


def resolve_url(repo, rev, path):
    return f"{HF}/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"


def resolve_ref(repo, ref):
    """Commit sha a ref (e.g. refs/convert/parquet) points to."""
    with _req(f"{HF}/api/datasets/{repo}/refs") as r:
        j = json.load(r)
    for group in j.values():
        for x in group:
            if x.get("ref") == ref:
                return x["targetCommit"]
    raise KeyError(ref)


def list_files(repo, rev, prefix=""):
    """All file paths under prefix at revision (follows tree-API pagination)."""
    url = f"{HF}/api/datasets/{repo}/tree/{urllib.parse.quote(rev, safe='')}/{urllib.parse.quote(prefix)}?recursive=true"
    out = []
    while url:
        with _req(url) as r:
            out += [f["path"] for f in json.load(r) if f["type"] == "file"]
            link = r.headers.get("Link") or ""
        url = None
        for part in link.split(","):
            if 'rel="next"' in part:
                url = part[part.index("<") + 1:part.index(">")]
    return sorted(out)


class HttpFile(io.RawIOBase):
    """Seekable read-only file over HTTP range requests (pyarrow reads footer + chosen column chunks)."""

    def __init__(self, url):
        self.orig, self.pos, self.fetched = url, 0, 0
        self._resolve()

    def _resolve(self):
        with _req(self.orig, {"Range": "bytes=0-0"}) as r:
            self.url = r.geturl()  # signed CDN url after redirect; expires, so re-resolved on errors
            self.size = int(r.headers["Content-Range"].split("/")[-1])

    def readable(self):
        return True

    def seekable(self):
        return True

    def tell(self):
        return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def read(self, n=-1):
        if n is None or n < 0:
            n = self.size - self.pos
        n = min(n, self.size - self.pos)
        if n <= 0:
            return b""
        for attempt in range(6):
            try:
                with _req(self.url, {"Range": f"bytes={self.pos}-{self.pos + n - 1}"}, tries=1) as r:
                    data = r.read()
                if len(data) != n:
                    raise ConnectionError(f"short read {len(data)}/{n}")
                break
            except Exception:
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
                try:
                    self._resolve()
                except Exception:
                    pass
        self.pos += n
        self.fetched += n
        return data

    def readinto(self, b):
        d = self.read(len(b))
        b[:len(d)] = d
        return len(d)


def parquet(repo, rev, path):
    import pyarrow.parquet as pq
    f = HttpFile(resolve_url(repo, rev, path))
    return pq.ParquetFile(f), f


def download(repo, rev, path, dest: Path):
    dest = Path(dest)
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    with _req(resolve_url(repo, rev, path), timeout=600) as r, open(tmp, "wb") as f:
        while True:
            b = r.read(1 << 20)
            if not b:
                break
            f.write(b)
    tmp.rename(dest)
    return dest
