import hashlib
from types import SimpleNamespace

import pytest

from scripts import download_zh_gapfill_20261102 as download


def test_range_resume_preserves_bytes_and_checks_final_upstream_hash(tmp_path, monkeypatch):
    payload = b'complete upstream bytes'
    monkeypatch.setitem(download.SOURCES, 'fixture', {
        'repo': 'fixture/source', 'revision': 'pinned', 'path': 'body.tsv',
        'bytes': len(payload), 'sha256': hashlib.sha256(payload).hexdigest(), 'small': [],
    })
    partial = tmp_path / 'fixture/ranges/000000000000/chunk.part'
    partial.parent.mkdir(parents=True)
    partial.write_bytes(payload[:5])
    calls = []

    class Response:
        status_code = 206
        headers = {'Content-Range': f'bytes 5-{len(payload)-1}/{len(payload)}'}

        def raise_for_status(self):
            pass

        def iter_content(self, size):
            yield payload[5:]

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def request(url, **kwargs):
        calls.append(kwargs['headers']['Range'])
        return Response()

    monkeypatch.setattr(download.requests, 'Session', lambda: SimpleNamespace(get=request, close=lambda: None))
    download.acquire(tmp_path, 'fixture', 1)
    assert calls == [f'bytes=5-{len(payload)-1}']
    assert (tmp_path / 'fixture/body.tsv').read_bytes() == payload
    # A sealed output is reusable without another request, but tampering fails.
    download.acquire(tmp_path, 'fixture', 1)
    assert len(calls) == 1
    (tmp_path / 'fixture/body.tsv').write_bytes(b'wrong')
    with pytest.raises(ValueError, match='pinned upstream'):
        download.acquire(tmp_path, 'fixture', 1)


def test_non_range_response_never_gets_appended(tmp_path, monkeypatch):
    monkeypatch.setitem(download.SOURCES, 'fixture', {
        'repo': 'fixture/source', 'revision': 'pinned', 'path': 'body.tsv',
        'bytes': 4, 'sha256': hashlib.sha256(b'test').hexdigest(), 'small': [],
    })

    class Response:
        status_code = 200
        headers = {}

        def raise_for_status(self):
            pass

    monkeypatch.setattr(download.requests, 'Session', lambda: SimpleNamespace(
        get=lambda *args, **kwargs: Response(), close=lambda: None))
    with pytest.raises(ValueError, match='exact requested byte range'):
        download.acquire(tmp_path, 'fixture', 1)
    assert not (tmp_path / 'fixture/body.tsv').exists()
    assert not (tmp_path / 'fixture/ranges/000000000000/chunk.part').exists()
