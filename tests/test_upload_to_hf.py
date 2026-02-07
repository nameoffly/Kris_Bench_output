import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import upload_to_hf as u


class _FakeApi:
    def __init__(self):
        self.calls = []

    def repo_info(self, **kwargs):
        self.calls.append(("repo_info", kwargs))
        return {"ok": True}

    def create_repo(self, **kwargs):
        self.calls.append(("create_repo", kwargs))
        return {"created": True}

    def upload_file(self, **kwargs):
        self.calls.append(("upload_file", kwargs))
        return "https://huggingface.co/fake/commit/123"


class UploadToHfTests(unittest.TestCase):
    def test_resolve_token_prefers_cli_token(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=True):
            self.assertEqual(u.resolve_token("cli-token"), "cli-token")

    def test_resolve_token_reads_env_token(self):
        with mock.patch.dict(os.environ, {"HF_TOKEN": "env-token"}, clear=True):
            self.assertEqual(u.resolve_token(None), "env-token")

    def test_resolve_token_raises_without_token(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(ValueError):
                u.resolve_token(None)

    def test_validate_local_file_rejects_missing_path(self):
        with self.assertRaises(FileNotFoundError):
            u.validate_local_file("/tmp/path-does-not-exist-12345.zip")

    def test_validate_local_file_rejects_empty_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "empty.zip"
            fp.write_bytes(b"")
            with self.assertRaises(ValueError):
                u.validate_local_file(str(fp))

    def test_ensure_repo_exists_checks_repo_info_by_default(self):
        api = _FakeApi()
        u.ensure_repo_exists(
            api=api,
            repo_id="alice/repo",
            repo_type="dataset",
            create_repo=False,
            private=False,
            token="token",
        )
        self.assertEqual(api.calls[0][0], "repo_info")
        self.assertEqual(api.calls[0][1]["repo_id"], "alice/repo")

    def test_ensure_repo_exists_creates_repo_when_enabled(self):
        api = _FakeApi()
        u.ensure_repo_exists(
            api=api,
            repo_id="alice/repo",
            repo_type="dataset",
            create_repo=True,
            private=True,
            token="token",
        )
        self.assertEqual(api.calls[0][0], "create_repo")
        self.assertEqual(api.calls[0][1]["repo_id"], "alice/repo")
        self.assertTrue(api.calls[0][1]["private"])

    def test_upload_zip_calls_hf_api(self):
        api = _FakeApi()
        with tempfile.TemporaryDirectory() as tmp:
            fp = Path(tmp) / "output_bagel.zip"
            fp.write_bytes(b"zip-content")
            url = u.upload_zip(
                api=api,
                local_file=str(fp),
                repo_id="alice/repo",
                path_in_repo="bagel/output_bagel.zip",
                repo_type="dataset",
                commit_message="upload test zip",
                token="token",
            )

        self.assertIn("huggingface.co", url)
        self.assertEqual(api.calls[0][0], "upload_file")
        self.assertEqual(api.calls[0][1]["repo_id"], "alice/repo")
        self.assertEqual(api.calls[0][1]["path_in_repo"], "bagel/output_bagel.zip")


if __name__ == "__main__":
    unittest.main()
