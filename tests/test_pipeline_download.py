import hashlib
import http.server
import io
import socketserver
import tempfile
import threading
import unittest
import zipfile
from pathlib import Path

from dci_bench.data.pipeline.download import (
    ArchiveSecurityError,
    IntegrityError,
    SnapshotConflictError,
    copy_file,
    download_url,
    nonblocking_file_lock,
    safe_extract_archive,
)


class _FixtureHandler(http.server.BaseHTTPRequestHandler):
    payload = b"downloaded payload"
    requests = 0
    mode = "normal"

    def do_GET(self):  # noqa: N802
        type(self).requests += 1
        body = type(self).payload
        if self.path == "/missing":
            self.send_response(404)
            self.end_headers()
            return
        if self.path == "/retry" and type(self).requests == 1:
            self.send_response(200)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body[:4])
            self.wfile.flush()
            self.connection.shutdown(1)
            return
        self.send_response(200)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_args):
        return


class DownloadTest(unittest.TestCase):
    def setUp(self):
        _FixtureHandler.requests = 0
        self.server = socketserver.ThreadingTCPServer(("127.0.0.1", 0), _FixtureHandler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)

    def test_retry_detects_truncated_content_length_and_replaces_atomically(self):
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "payload.bin"
            digest, size = download_url(
                self.base + "/retry",
                destination,
                retries=2,
                expected_sha256=hashlib.sha256(_FixtureHandler.payload).hexdigest(),
            )
            self.assertEqual(destination.read_bytes(), _FixtureHandler.payload)
            self.assertEqual(size, len(_FixtureHandler.payload))
            self.assertEqual(digest, hashlib.sha256(_FixtureHandler.payload).hexdigest())
            self.assertGreaterEqual(_FixtureHandler.requests, 2)
            self.assertFalse(destination.with_name("payload.bin.partial").exists())

    def test_size_and_hash_failures_leave_partial_bytes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(IntegrityError):
                download_url(self.base + "/payload", root / "too-large", max_bytes=3, retries=0)
            with self.assertRaises(IntegrityError):
                download_url(self.base + "/payload", root / "bad-hash", expected_sha256="0" * 64, retries=0)
            self.assertTrue((root / "too-large.partial").exists())
            self.assertTrue((root / "bad-hash.partial").exists())

    def test_copy_is_bytes_not_a_symlink_and_completed_file_is_not_overwritten(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "source.txt"
            source.write_bytes(b"source")
            link = root / "link.txt"
            link.symlink_to(source)
            destination = root / "out.txt"
            copy_file(link, destination)
            self.assertFalse(destination.is_symlink())
            self.assertEqual(destination.read_bytes(), b"source")
            with self.assertRaises(FileExistsError):
                copy_file(source, destination)

    def test_archive_traversal_and_links_are_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            traversal = root / "traversal.zip"
            with zipfile.ZipFile(traversal, "w") as archive:
                archive.writestr("../escape.txt", b"escape")
            with self.assertRaises(ArchiveSecurityError):
                safe_extract_archive(traversal, root / "out")
            self.assertTrue((root / "out.partial").is_dir())

            symlink = root / "symlink.zip"
            info = zipfile.ZipInfo("link")
            info.external_attr = (0o120777 << 16) | 0xA000
            with zipfile.ZipFile(symlink, "w") as archive:
                archive.writestr(info, b"/tmp/target")
            with self.assertRaises(ArchiveSecurityError):
                safe_extract_archive(symlink, root / "link-out")

    def test_snapshot_lock_is_nonblocking(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / "snapshot.lock"
            with nonblocking_file_lock(lock):
                with self.assertRaises(SnapshotConflictError):
                    with nonblocking_file_lock(lock):
                        pass
