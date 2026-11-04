# Copyright 2024 ByteDance and/or its affiliates.
# Modifications Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# This file has been modified by Molence Bio
# Based on Protenix: https://github.com/bytedance/Protenix
# Upstream revision: 0ae893499c7fc039c102bc7a70173d4c394381d5
# Upstream file: tests/test_msa_service_retries.py

"""Offline transport retry checks for every ColabFold request stage."""

import contextlib
import io
import tarfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from mol.web_service import colab_request_utils as writer


STAGES = ("submit", "status", "download")
TRANSPORT_ERRORS = (
    writer.requests.exceptions.ConnectionError,
    writer.requests.exceptions.Timeout,
)


class RetryServer:
    def __init__(self, stage=None, errors=()):
        self.stage = stage
        self.errors = list(errors)
        self.calls = dict.fromkeys(STAGES, 0)
        query = b">query_0\nACDEFG\n"
        files = {"bfd.mgnify30.metaeuk30.smag30.a3m": query, "uniref.a3m": query}
        self.urls = []
        stream = io.BytesIO()
        with tarfile.open(fileobj=stream, mode="w:gz") as handle:
            for name, data in files.items():
                entry = tarfile.TarInfo(name)
                entry.size = len(data)
                handle.addfile(entry, io.BytesIO(data))
        self.archive = stream.getvalue()

    def attempt(self, stage):
        self.calls[stage] += 1
        if stage == self.stage and self.errors:
            raise self.errors.pop(0)

    def post(self, url, **kwargs):
        self.urls.append(url)
        self.attempt("submit")
        response = Mock()
        response.json.return_value = {
            "status": "RUNNING" if self.stage == "status" else "COMPLETE",
            "id": "synthetic",
        }
        return response

    def get(self, url, **kwargs):
        self.urls.append(url)
        if url.endswith("/result/download/synthetic"):
            self.attempt("download")
            return Mock(content=self.archive)
        if not url.endswith("/ticket/synthetic"):
            raise AssertionError(f"Unexpected HTTP request: {url}")
        self.attempt("status")
        response = Mock()
        response.json.return_value = {"status": "COMPLETE", "id": "synthetic"}
        return response

    @contextlib.contextmanager
    def active(self):
        with patch.object(writer.requests, "post", side_effect=self.post), patch.object(
            writer.requests, "get", side_effect=self.get
        ), patch.object(writer.time, "sleep") as sleep, contextlib.redirect_stdout(
            io.StringIO()
        ), contextlib.redirect_stderr(
            io.StringIO()
        ):
            yield sleep

    def run(self, root, host_url="https://offline.invalid"):
        return writer.run_mmseqs2_service(
            ">query_0\nACDEFG",
            str(root / "msa"),
            host_url=host_url,
            user_agent="msa-retry-test",
        )


class TestMSAServiceRetries(unittest.TestCase):
    @staticmethod
    def retry_sleeps(sleep):
        return [call.args[0] for call in sleep.call_args_list if call.args[0] == 5]

    def test_successful_requests_need_no_retries(self):
        with tempfile.TemporaryDirectory() as directory:
            server = RetryServer()
            with server.active() as sleep:
                result = server.run(Path(directory))
            self.assertEqual(server.calls, dict(submit=1, status=0, download=1))
            self.assertEqual(self.retry_sleeps(sleep), [])
            self.assertTrue((Path(result) / "non_pairing.a3m").is_file())

    def test_transient_transport_errors_recover(self):
        self.assert_success_after_errors(2)

    def test_success_on_the_last_allowed_attempt(self):
        self.assert_success_after_errors(5)

    def assert_success_after_errors(self, count):
        for stage in STAGES:
            for error_type in TRANSPORT_ERRORS:
                with self.subTest(
                    stage=stage, error=error_type.__name__
                ), tempfile.TemporaryDirectory() as directory:
                    errors = [
                        error_type("synthetic transport error") for _ in range(count)
                    ]
                    server = RetryServer(stage, errors)
                    with server.active() as sleep:
                        server.run(Path(directory))
                    self.assertEqual(server.calls[stage], count + 1)
                    self.assertEqual(self.retry_sleeps(sleep), [5] * count)

    def test_transport_errors_exhaust_the_retry_budget(self):
        for stage in STAGES:
            for error_type in TRANSPORT_ERRORS:
                with self.subTest(
                    stage=stage, error=error_type.__name__
                ), tempfile.TemporaryDirectory() as directory:
                    errors = [error_type("synthetic transport error") for _ in range(6)]
                    server = RetryServer(stage, errors)
                    with server.active() as sleep, self.assertRaises(
                        error_type
                    ) as raised:
                        server.run(Path(directory))
                    self.assertIs(raised.exception, errors[-1])
                    self.assertEqual(server.calls[stage], 6)
                    self.assertEqual(self.retry_sleeps(sleep), [5] * 5)

    def test_mixed_transport_errors_share_one_budget(self):
        for stage in STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                errors = [
                    error("synthetic transport error")
                    for _ in range(3)
                    for error in TRANSPORT_ERRORS
                ]
                server = RetryServer(stage, errors)
                with server.active() as sleep, self.assertRaises(
                    writer.requests.exceptions.RequestException
                ) as raised:
                    server.run(Path(directory))
                self.assertIs(raised.exception, errors[-1])
                self.assertEqual(server.calls[stage], 6)
                self.assertEqual(self.retry_sleeps(sleep), [5] * 5)

    def test_non_transport_errors_propagate_immediately(self):
        for stage in STAGES:
            with self.subTest(stage=stage), tempfile.TemporaryDirectory() as directory:
                server = RetryServer(stage, [ValueError("synthetic invalid argument")])
                with server.active() as sleep, self.assertRaisesRegex(
                    ValueError, "synthetic invalid argument"
                ):
                    server.run(Path(directory))
                self.assertEqual(server.calls[stage], 1)
                self.assertEqual(self.retry_sleeps(sleep), [])

    def test_service_requires_an_explicit_host(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(writer.requests, "post") as post, patch.object(
                writer.requests, "get"
            ) as get, self.assertRaisesRegex(ValueError, "host_url"):
                writer.run_mmseqs2_service(
                    ">query_0\nACDEFG", str(Path(directory) / "msa")
                )
            post.assert_not_called()
            get.assert_not_called()

    def test_service_normalizes_a_trailing_host_slash(self):
        with tempfile.TemporaryDirectory() as directory:
            server = RetryServer()
            with server.active():
                server.run(Path(directory), host_url="https://offline.invalid/")
            self.assertEqual(server.urls[0], "https://offline.invalid/ticket/msa")
            self.assertEqual(
                server.urls[1], "https://offline.invalid/result/download/synthetic"
            )


if __name__ == "__main__":
    unittest.main()
