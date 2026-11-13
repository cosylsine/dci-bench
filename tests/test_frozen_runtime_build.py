"""Check the shipped OCI identity describes the exact offline rootfs bytes."""
import gzip
import hashlib
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

from scripts.build_frozen_runtime import write_oci_layout


class FrozenImageTests(unittest.TestCase):
    def test_image_platform_descriptors_and_layer_match_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            bundle = root / 'rootfs.tar'
            with tarfile.open(bundle, 'w') as archive:
                entry = tarfile.TarInfo('usr')
                entry.type = tarfile.DIRTYPE
                entry.mode = 0o555
                archive.addfile(entry)
            image_digests, archive_hashes = [], []
            for name in ('first', 'second'):
                output = root / name
                output.mkdir()
                image_digest = write_oci_layout(bundle, output)
                image_digests.append(image_digest)
                archive_path = output / 'runtime-image.tar'
                archive_hashes.append(hashlib.sha256(archive_path.read_bytes()).hexdigest())
                with tarfile.open(archive_path) as archive:
                    def read_blob(descriptor):
                        data = archive.extractfile('blobs/' + descriptor['digest'].replace(':', '/')).read()
                        self.assertEqual(len(data), descriptor['size'])
                        self.assertEqual('sha256:' + hashlib.sha256(data).hexdigest(), descriptor['digest'])
                        return data

                    layout = json.load(archive.extractfile('oci-layout'))
                    self.assertEqual(layout['imageLayoutVersion'], '1.0.0')
                    index = json.load(archive.extractfile('index.json'))
                    descriptor = index['manifests'][0]
                    self.assertEqual(descriptor['digest'], image_digest)
                    self.assertEqual(descriptor['platform'], {'os': 'linux', 'architecture': 'amd64'})
                    manifest = json.loads(read_blob(descriptor))
                    config = json.loads(read_blob(manifest['config']))
                    self.assertEqual((config['os'], config['architecture']), ('linux', 'amd64'))
                    layer = gzip.decompress(read_blob(manifest['layers'][0]))
                    self.assertEqual(layer, bundle.read_bytes())
                    self.assertEqual(config['rootfs']['diff_ids'], ['sha256:' + hashlib.sha256(layer).hexdigest()])
            self.assertEqual(image_digests[0], image_digests[1])
            self.assertEqual(archive_hashes[0], archive_hashes[1])
