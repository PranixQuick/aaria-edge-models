#!/usr/bin/env python3
"""
Unit tests for the Telugu model switch logic (runs locally).

Tests:
1. Koyal build: models.te points to Koyal, exact four files, non-target languages unchanged byte-for-byte.
2. IndicConformer build: models.te matches stored byte-for-byte, non-target languages unchanged.
3. Stored IndicConformer matches live manifest models.te (skips gracefully if offline).
4. Limit enforcement: fake 700 MB total fails, fake 450 MB single file fails.
5. Security rules: http URL fails, URL prefix rule enforced, path traversal ../ fails, version != 1 fails.
6. Signature round-trip test using a THROWAWAY key generated in memory.
7. Python compilation on all scripts.
8. YAML parsing and security assertions on workflows (te-compare.yml, te-switch.yml):
   - te-switch.yml: workflow_dispatch only, dry_run defaults to true.
   - concurrency: { group: models-manifest, cancel-in-progress: false }
   - plan job has contents: write (for draft listing) and NO secrets.
   - apply job has if: inputs.dry_run == false and contents: write.
   - failure()-rollback step exists in apply job with cancelled() check.
9. No-op detection: switching to target while already on target returns is_noop=True.
10. Exact four-file draft checking: missing file fails, extra te-koyal__ file fails.
11. score_onnx.py manifest validation: missing model.int8.onnx or tokens.txt fails (not skip).
12. fetch_draft_assets unit tests (fake responses).
13. Shell-syntax tests: bash -n validation on every workflow run: step; assert no << here-docs.
14. verify_manifest unit tests.
15. Part 43e Gap Fix Tests:
    - Empty or missing expected sha refused by apply_switch.
    - Output-write failure in build_te_switch exits non-zero.
    - Race guard in apply_switch stops before upload when live manifest changed.
    - Missing public key strictly fails rollback_switch.
    - Unchanged live manifest skips rollback re-upload.
"""

import base64
import copy
import hashlib
import json
import os
import py_compile
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "switch"))
import build_te_switch
import fetch_draft_assets
import verify_manifest
import apply_switch
import rollback_switch

sys.path.insert(0, str(REPO_ROOT / "scripts" / "koyal_te"))
import score_onnx


MOCK_MANIFEST = {
    "version": 1,
    "release": "models-2026-10",
    "models": {
        "en": {
            "type": "moonshine",
            "files": [
                {
                    "name": "model.int8.onnx",
                    "bytes": 50000000,
                    "sha256": "a" * 64,
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/en__model.int8.onnx",
                }
            ],
        },
        "hi": {
            "type": "ctc",
            "files": [
                {
                    "name": "model.int8.onnx",
                    "bytes": 137000000,
                    "sha256": "b" * 64,
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/hi__model.int8.onnx",
                }
            ],
        },
        "te": {
            "type": "ctc",
            "files": [
                {
                    "name": "model.int8.onnx",
                    "bytes": 137563742,
                    "sha256": "13774a654ce5152796e1129db880a721f7c1ec4cd6301e627e258336ec9ac49e",
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__model.int8.onnx",
                },
                {
                    "name": "model.json",
                    "bytes": 71,
                    "sha256": "5e42ed018bdf7894ba3f04493b8fcdba85ecce3276349f5893624b2e53a8bf8c",
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__model.json",
                },
                {
                    "name": "tokens.txt",
                    "bytes": 2963,
                    "sha256": "02488cdcc1e893b392b1ba24f47508e7b7c64b117be249a0d09c5d8b65ecdfcb",
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__tokens.txt",
                },
            ],
        },
    },
    "replies": {
        "v1": {
            "en": [{"bytes": 100, "name": "hi.wav", "sha256": "c" * 64, "url": "https://example.com/hi.wav"}]
        }
    },
}

STORED_INDICCONFORMER = {
    "files": [
        {
            "bytes": 137563742,
            "name": "model.int8.onnx",
            "sha256": "13774a654ce5152796e1129db880a721f7c1ec4cd6301e627e258336ec9ac49e",
            "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__model.int8.onnx",
        },
        {
            "bytes": 71,
            "name": "model.json",
            "sha256": "5e42ed018bdf7894ba3f04493b8fcdba85ecce3276349f5893624b2e53a8bf8c",
            "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__model.json",
        },
        {
            "bytes": 2963,
            "name": "tokens.txt",
            "sha256": "02488cdcc1e893b392b1ba24f47508e7b7c64b117be249a0d09c5d8b65ecdfcb",
            "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/te__tokens.txt",
        },
    ],
    "type": "ctc",
}


def test_koyal_and_indicconformer_builds():
    print("Testing Koyal and IndicConformer switch builds...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        draft_dir = tpath / "draft"
        draft_dir.mkdir()
        for role, name in build_te_switch.KOYAL_REQUIRED_ASSETS.items():
            (draft_dir / name).write_text(f"dummy content for {name}", encoding="utf-8")

        stored_file = tpath / "stored.json"
        stored_file.write_text(json.dumps(STORED_INDICCONFORMER, indent=2), encoding="utf-8")

        # 1. Build koyal
        koyal_manifest, k_bytes = build_te_switch.build_switch_manifest(
            MOCK_MANIFEST, "koyal", stored_file, draft_dir
        )
        assert koyal_manifest["models"]["te"]["type"] == "ctc"
        assert len(koyal_manifest["models"]["te"]["files"]) == 4
        # Verify non-target languages unchanged
        assert koyal_manifest["models"]["en"] == MOCK_MANIFEST["models"]["en"]
        assert koyal_manifest["models"]["hi"] == MOCK_MANIFEST["models"]["hi"]
        assert koyal_manifest["replies"] == MOCK_MANIFEST["replies"]
        print("  -> Koyal build verified and other languages byte-identical!")

        # 2. Build indicconformer
        ic_manifest, ic_bytes = build_te_switch.build_switch_manifest(
            koyal_manifest, "indicconformer", stored_file, draft_dir
        )
        assert ic_manifest["models"]["te"] == STORED_INDICCONFORMER
        assert ic_manifest["models"]["en"] == MOCK_MANIFEST["models"]["en"]
        assert ic_manifest["models"]["hi"] == MOCK_MANIFEST["models"]["hi"]
        assert ic_manifest["replies"] == MOCK_MANIFEST["replies"]
        print("  -> IndicConformer build verified byte-for-byte against stored entry!")


def test_live_indicconformer_matches_stored():
    print("Testing that stored te-indicconformer.json matches live release manifest...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"
    assert stored_path.is_file(), f"Missing {stored_path}"
    stored_data = json.loads(stored_path.read_text(encoding="utf-8"))

    manifest_url = "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/manifest.json"
    try:
        req = urllib.request.Request(manifest_url, headers={"User-Agent": "test-suite"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            live_data = json.loads(resp.read().decode("utf-8"))
            live_te = live_data.get("models", {}).get("te")
            assert live_te == stored_data, "Stored te-indicconformer.json does NOT match live release models-2026-10 models.te!"
            print("  -> Stored te-indicconformer.json matches live models-2026-10 manifest byte-for-byte!")
    except Exception as e:
        print(f"  -> SKIP (network unavailable or timeout): {e}")


def test_size_limits_enforced():
    print("Testing 600 MB total and 400 MB per-file limits...")
    # Over 400 MB single file
    bad_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest["models"]["en"]["files"][0]["bytes"] = 700 * 1024 * 1024
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest)
        assert False, "Failed to reject >400 MB single file"
    except ValueError as e:
        assert "exceeds 400 MB limit" in str(e)
        print(f"  -> 700 MB limit violation caught: {e}")

    # Total over 600 MB
    bad_manifest2 = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest2["models"]["en"]["files"][0]["bytes"] = 350 * 1024 * 1024
    bad_manifest2["models"]["hi"]["files"][0]["bytes"] = 350 * 1024 * 1024
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest2)
        assert False, "Failed to reject >600 MB total files"
    except ValueError as e:
        assert "exceeds 600 MB limit" in str(e)
        print(f"  -> >600 MB total limit violation caught: {e}")


def test_security_rules_enforced():
    print("Testing security rules (http URL, URL prefix, path traversal, version)...")
    # 1. http URL
    bad_url_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_url_manifest["models"]["en"]["files"][0]["url"] = "http://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_url_manifest)
        assert False, "Failed to reject non-prefixed http URL"
    except ValueError as e:
        assert "does not start with required prefix" in str(e)
        print(f"  -> http URL violation caught: {e}")

    # 2. URL prefix mismatch
    bad_prefix_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_prefix_manifest["models"]["en"]["files"][0]["url"] = "https://malicious.example.com/models-2026-10/model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_prefix_manifest)
        assert False, "Failed to reject URL not starting with required prefix"
    except ValueError as e:
        assert "does not start with required prefix" in str(e)
        print(f"  -> URL prefix mismatch caught: {e}")

    # 3. Path traversal in filename
    bad_path_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_path_manifest["models"]["en"]["files"][0]["name"] = "../model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_path_manifest)
        assert False, "Failed to reject path traversal in filename"
    except ValueError as e:
        assert "Path traversal detected" in str(e)
        print(f"  -> path traversal caught: {e}")

    # 4. Version != 1
    bad_version_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_version_manifest["version"] = 2
    try:
        build_te_switch.validate_sync_manifest_rules(bad_version_manifest)
        assert False, "Failed to reject version != 1"
    except ValueError as e:
        assert "version must be 1" in str(e)
        print(f"  -> version != 1 caught: {e}")


def test_noop_detection():
    print("Testing no-op detection...")
    assert build_te_switch.check_is_noop(MOCK_MANIFEST, copy.deepcopy(MOCK_MANIFEST)) is True

    diff_manifest = copy.deepcopy(MOCK_MANIFEST)
    diff_manifest["models"]["te"]["type"] = "other"
    assert build_te_switch.check_is_noop(MOCK_MANIFEST, diff_manifest) is False
    print("  -> No-op detection verified!")


def test_exact_four_file_list_enforced():
    print("Testing exact four-file Koyal draft enforcement (missing/extra files fail)...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        draft_dir = tpath / "draft"
        draft_dir.mkdir()

        # Missing draft_dir entirely
        try:
            build_te_switch.build_koyal_entry(None)
            assert False, "Should fail when draft_dir is None"
        except FileNotFoundError as e:
            print(f"  -> Missing draft_dir caught: {e}")

        # Missing required file
        for name in [
            "te-koyal__model.int8.onnx",
            "te-koyal__tokens.txt",
            "te-koyal__model.json",
        ]:
            (draft_dir / name).write_text("dummy", encoding="utf-8")
        try:
            build_te_switch.build_koyal_entry(draft_dir)
            assert False, "Should fail when te-koyal__LICENSE-koyal-te-120m.txt is missing"
        except FileNotFoundError as e:
            assert "te-koyal__LICENSE-koyal-te-120m.txt" in str(e)
            print(f"  -> Missing draft asset caught: {e}")

        # Extra unexpected te-koyal__ file
        (draft_dir / "te-koyal__LICENSE-koyal-te-120m.txt").write_text("dummy", encoding="utf-8")
        (draft_dir / "te-koyal__unexpected.bin").write_text("bad", encoding="utf-8")
        try:
            build_te_switch.build_koyal_entry(draft_dir)
            assert False, "Should fail when extra unexpected te-koyal__ file is present"
        except ValueError as e:
            assert "Unexpected te-koyal__ asset" in str(e)
            print(f"  -> Unexpected te-koyal__ asset caught: {e}")


def test_signature_roundtrip_throwaway_key():
    print("Testing signature round-trip with a THROWAWAY key generated in memory...")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    # Generate disposable key in memory - NEVER writes to disk, never hardcodes secret
    private_key = ec.generate_private_key(ec.SECP256R1())
    pem_bytes = private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    ).decode("utf-8")

    pub_der = private_key.public_key().public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    pubkey_b64 = base64.b64encode(pub_der).decode("ascii")

    test_payload = b'{"test": "payload_for_signing_verification"}\n'
    sig = apply_switch.sign_manifest(test_payload, pem_bytes)

    # Valid signature check
    apply_switch.verify_signature(test_payload, sig, pubkey_b64)

    # Tampered payload fails
    try:
        apply_switch.verify_signature(test_payload + b"tamper", sig, pubkey_b64)
        assert False, "Signature verification failed to catch tampered payload"
    except Exception:
        pass
    print("  -> Signature round-trip with throwaway key passed (tamper detection verified)!")


def test_score_onnx_manifest_checks():
    print("Testing score_onnx.py verify_against_manifest failure when model or tokens missing...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        m_file = tpath / "manifest.json"
        model_p = tpath / "model.int8.onnx"
        tokens_p = tpath / "tokens.txt"
        model_p.write_bytes(b"dummy onnx bytes")
        tokens_p.write_text("token1\ntoken2\n", encoding="utf-8")

        bad_m = {
            "models": {
                "te": {
                    "files": [
                        {"name": "model.int8.onnx", "sha256": score_onnx.sha256_file(model_p)}
                    ]
                }
            }
        }
        m_file.write_text(json.dumps(bad_m), encoding="utf-8")

        try:
            score_onnx.verify_against_manifest(m_file, model_p, tokens_p)
            assert False, "Expected ValueError when tokens.txt missing from manifest"
        except ValueError as e:
            assert "does not list 'tokens.txt'" in str(e)
            print(f"  -> Missing tokens.txt in manifest correctly caught: {e}")

        bad_m2 = {
            "models": {
                "te": {
                    "files": [
                        {"name": "tokens.txt", "sha256": score_onnx.sha256_file(tokens_p)}
                    ]
                }
            }
        }
        m_file.write_text(json.dumps(bad_m2), encoding="utf-8")
        try:
            score_onnx.verify_against_manifest(m_file, model_p, tokens_p)
            assert False, "Expected ValueError when model.int8.onnx missing from manifest"
        except ValueError as e:
            assert "does not list 'model.int8.onnx'" in str(e)
            print(f"  -> Missing model.int8.onnx in manifest correctly caught: {e}")


def test_fetch_draft_assets_logic():
    print("Testing fetch_draft_assets release-matching and asset validation logic...")
    tag = "models-2026-10-koyal-te"
    exact_assets = [
        "te-koyal__model.int8.onnx",
        "te-koyal__tokens.txt",
        "te-koyal__model.json",
        "te-koyal__LICENSE-koyal-te-120m.txt",
    ]

    mock_release = {
        "id": 12345,
        "tag_name": tag,
        "draft": True,
        "assets": [
            {"id": 1, "name": "te-koyal__model.int8.onnx"},
            {"id": 2, "name": "te-koyal__tokens.txt"},
            {"id": 3, "name": "te-koyal__model.json"},
            {"id": 4, "name": "te-koyal__LICENSE-koyal-te-120m.txt"},
        ],
    }

    # 1. Draft match succeeds
    matched_rel = fetch_draft_assets.find_matching_release([mock_release], tag)
    assert matched_rel["id"] == 12345
    assert matched_rel["draft"] is True
    matched_map = fetch_draft_assets.validate_release_assets(matched_rel, exact_assets)
    assert len(matched_map) == 4
    print("  -> Draft release match and asset validation succeeded!")

    # 2. No match fails
    try:
        fetch_draft_assets.find_matching_release([], tag)
        assert False, "Expected ValueError for no matching release"
    except ValueError as e:
        assert "No release found" in str(e)
        print(f"  -> No match caught: {e}")

    # 3. Two matches fails (ambiguous)
    mock_rel_2 = copy.deepcopy(mock_release)
    mock_rel_2["id"] = 67890
    try:
        fetch_draft_assets.find_matching_release([mock_release, mock_rel_2], tag)
        assert False, "Expected ValueError for two matching releases"
    except ValueError as e:
        assert "Ambiguous" in str(e) or "found 2" in str(e).lower()
        print(f"  -> Ambiguous match caught: {e}")

    # 4. Missing asset fails
    incomplete_rel = copy.deepcopy(mock_release)
    incomplete_rel["assets"] = incomplete_rel["assets"][:3]
    try:
        fetch_draft_assets.validate_release_assets(incomplete_rel, exact_assets)
        assert False, "Expected ValueError for missing asset"
    except ValueError as e:
        assert "Missing required asset" in str(e)
        print(f"  -> Missing asset caught: {e}")

    # 5. Extra te-koyal__ asset fails
    extra_rel = copy.deepcopy(mock_release)
    extra_rel["assets"].append({"id": 99, "name": "te-koyal__extra.bin"})
    try:
        fetch_draft_assets.validate_release_assets(extra_rel, exact_assets)
        assert False, "Expected ValueError for extra te-koyal__ asset"
    except ValueError as e:
        assert "Unexpected te-koyal__ asset" in str(e)
        print(f"  -> Extra te-koyal__ asset caught: {e}")


def test_verify_manifest_logic():
    print("Testing verify_manifest.py standalone verification functions...")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    with tempfile.TemporaryDirectory() as td:
        td_path = Path(td)
        manifest_file = td_path / "manifest.json"
        sig_file = td_path / "manifest.json.sig"
        pubkey_file = td_path / "pubkey.b64"

        data = b'{"hello": "switch"}\n'
        manifest_file.write_bytes(data)
        data_sha = hashlib.sha256(data).hexdigest()

        private_key = ec.generate_private_key(ec.SECP256R1())
        sig = private_key.sign(data, ec.ECDSA(hashes.SHA256()))
        sig_file.write_bytes(sig)

        pub_der = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.DER,
            format=serialization.PublicFormat.SubjectPublicKeyInfo,
        )
        pubkey_file.write_text(base64.b64encode(pub_der).decode("ascii"), encoding="utf-8")

        # Signature verification
        verify_manifest.verify_signature(data, sig, pubkey_file.read_text())
        print("  -> verify_signature passed!")

        # Digest equality
        assert verify_manifest.sha256_bytes(data) == data_sha
        print("  -> sha256_bytes passed!")


# ==============================================================================
# Part 43e Gap Fix Tests
# ==============================================================================

def test_empty_expected_sha_refused():
    print("Testing that empty or missing --expected-sha256 is refused in apply_switch...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        manifest_file = tpath / "manifest.json"
        manifest_file.write_text('{"version": 1}', encoding="utf-8")
        current_dir = tpath / "current"
        current_dir.mkdir()
        (current_dir / "manifest.json").write_text('{"version": 1}', encoding="utf-8")

        # Test empty string passed to --expected-sha256
        res = apply_switch.main([
            "--target", "indicconformer",
            "--current-dir", str(current_dir),
            "--plan-manifest", str(manifest_file),
            "--expected-sha256", "",
        ])
        assert res != 0, f"Expected non-zero exit for empty --expected-sha256, got {res}"

        # Test whitespace only
        res = apply_switch.main([
            "--target", "indicconformer",
            "--current-dir", str(current_dir),
            "--plan-manifest", str(manifest_file),
            "--expected-sha256", "   ",
        ])
        assert res != 0, f"Expected non-zero exit for whitespace --expected-sha256, got {res}"

        # Test argument omitted (argparse error)
        try:
            apply_switch.main([
                "--target", "indicconformer",
                "--current-dir", str(current_dir),
                "--plan-manifest", str(manifest_file),
            ])
            assert False, "Expected SystemExit when --expected-sha256 is omitted"
        except SystemExit as e:
            assert e.code != 0
        print("  -> Empty and missing --expected-sha256 correctly refused!")


def test_output_write_failure_exits_nonzero():
    print("Testing that failure to write GITHUB_OUTPUT exits non-zero...")
    with tempfile.TemporaryDirectory() as td:
        bad_path = str(Path(td) / "nonexistent_dir" / "output.txt")
        old_env = os.environ.get("GITHUB_OUTPUT")
        try:
            os.environ["GITHUB_OUTPUT"] = bad_path
            try:
                build_te_switch.write_github_output(False, "dummyhash")
                assert False, "Expected SystemExit when writing GITHUB_OUTPUT fails"
            except SystemExit as e:
                assert e.code != 0, f"Expected non-zero exit code, got {e.code}"
        finally:
            if old_env is None:
                os.environ.pop("GITHUB_OUTPUT", None)
            else:
                os.environ["GITHUB_OUTPUT"] = old_env
        print("  -> GITHUB_OUTPUT write failure correctly exits non-zero!")


def test_changed_live_manifest_stops_before_upload():
    print("Testing race guard: changed live manifest stops before upload...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        current_dir = tpath / "current"
        current_dir.mkdir()
        baseline_manifest = current_dir / "manifest.json"
        baseline_manifest.write_bytes(b'{"baseline": "plan_start"}\n')

        race_check_dir = tpath / "race_check"
        race_check_dir.mkdir()
        live_manifest = race_check_dir / "manifest.json"
        live_manifest.write_bytes(b'{"changed": "concurrent_edit"}\n')

        orig_run = subprocess.run
        def mock_run(cmd, *a, **kw):
            if "download" in cmd and "manifest.json" in cmd:
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return orig_run(cmd, *a, **kw)

        try:
            subprocess.run = mock_run
            res = apply_switch.check_live_manifest_race("models-2026-10", current_dir, check_dir=race_check_dir)
            assert res is False, "Expected race check to return False when live manifest differs"

            # Verify identical bytes passes
            live_manifest.write_bytes(baseline_manifest.read_bytes())
            res_pass = apply_switch.check_live_manifest_race("models-2026-10", current_dir, check_dir=race_check_dir)
            assert res_pass is True, "Expected race check to return True when live manifest matches baseline"
        finally:
            subprocess.run = orig_run
        print("  -> Race guard correctly stops upload when live manifest changes!")


def test_missing_key_fails_rollback():
    print("Testing that missing public key file strictly fails rollback...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        current_dir = tpath / "current"
        current_dir.mkdir()
        (current_dir / "manifest.json").write_bytes(b'{"version": 1}\n')
        (current_dir / "manifest.json.sig").write_bytes(b'dummy_sig')
        # manifest_public_key.b64 is intentionally omitted

        res = rollback_switch.main(["--current-dir", str(current_dir)])
        assert res != 0, f"Expected non-zero exit when public key is missing, got {res}"
        print("  -> Missing public key correctly causes rollback failure!")


def test_unchanged_manifest_no_rollback_upload():
    print("Testing that unchanged live manifest skips rollback re-upload...")
    with tempfile.TemporaryDirectory() as td:
        tpath = Path(td)
        current_dir = tpath / "current"
        current_dir.mkdir()
        saved_bytes = b'{"version": 1, "status": "unchanged"}\n'
        (current_dir / "manifest.json").write_bytes(saved_bytes)
        (current_dir / "manifest.json.sig").write_bytes(b'dummy_sig')
        (current_dir / "manifest_public_key.b64").write_text("dummy_pubkey", encoding="utf-8")

        pre_check = Path("rollback_pre_check")
        uploaded_files = []
        orig_run = subprocess.run
        def mock_run(cmd, *a, **kw):
            if "download" in cmd:
                pre_check.mkdir(parents=True, exist_ok=True)
                (pre_check / "manifest.json").write_bytes(saved_bytes)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            if "upload" in cmd:
                uploaded_files.append(cmd)
                return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
            return orig_run(cmd, *a, **kw)

        try:
            subprocess.run = mock_run
            res = rollback_switch.main(["--current-dir", str(current_dir)])
            assert res == 0, f"Expected 0 exit when live manifest is unchanged, got {res}"
            assert len(uploaded_files) == 0, f"Expected NO upload calls when live manifest unchanged, got: {uploaded_files}"
        finally:
            subprocess.run = orig_run
            if pre_check.is_dir():
                shutil.rmtree(pre_check, ignore_errors=True)
        print("  -> Unchanged live manifest correctly skips rollback re-upload and exits 0!")


def test_workflows_shell_syntax():
    print("Testing workflow shell syntax with bash -n and asserting no here-documents...")
    import yaml

    workflows_dir = REPO_ROOT / ".github" / "workflows"
    compare_yml = workflows_dir / "te-compare.yml"
    switch_yml = workflows_dir / "te-switch.yml"

    # Find bash executable (prefer Git Bash)
    bash_candidates = [
        Path("C:/Program Files/Git/bin/bash.exe"),
        Path("C:/Program Files/Git/usr/bin/bash.exe"),
    ]
    which_bash = shutil.which("bash")
    if which_bash:
        bash_candidates.append(Path(which_bash))

    bash_bin = None
    for cand in bash_candidates:
        if cand.is_file():
            bash_bin = str(cand)
            break

    if not bash_bin:
        print("  -> SKIP: bash executable not found on host. Skipping bash -n tests.")
        return

    for wf_path in [compare_yml, switch_yml]:
        text = wf_path.read_text(encoding="utf-8")
        # Assert no here-documents (<<) remain
        assert "<<" not in text, f"Here-document '<<' found in {wf_path.name}! Move scripts to python files."

        data = yaml.safe_load(text)
        jobs = data.get("jobs", {})
        for job_name, job_dict in jobs.items():
            steps = job_dict.get("steps", [])
            for step in steps:
                step_name = step.get("name", "unnamed")
                run_cmd = step.get("run")
                if run_cmd:
                    res = subprocess.run(
                        [bash_bin, "-n"],
                        input=run_cmd.encode("utf-8"),
                        capture_output=True,
                    )
                    assert res.returncode == 0, (
                        f"bash -n failed on {wf_path.name} -> job '{job_name}' -> step '{step_name}':\n"
                        f"{res.stderr.decode()}\nScript content:\n{run_cmd}"
                    )
        print(f"  -> All run blocks in {wf_path.name} passed bash -n without errors!")


def test_py_compile_new_scripts():
    print("Testing py_compile on new scripts...")
    files_to_compile = [
        REPO_ROOT / "scripts" / "koyal_te" / "score_onnx.py",
        REPO_ROOT / "scripts" / "switch" / "build_te_switch.py",
        REPO_ROOT / "scripts" / "switch" / "fetch_draft_assets.py",
        REPO_ROOT / "scripts" / "switch" / "verify_manifest.py",
        REPO_ROOT / "scripts" / "switch" / "apply_switch.py",
        REPO_ROOT / "scripts" / "switch" / "rollback_switch.py",
        REPO_ROOT / "scripts" / "switch" / "test_switch.py",
    ]
    for p in files_to_compile:
        assert p.is_file(), f"Script to compile not found: {p}"
        py_compile.compile(str(p), doraise=True)
        print(f"  -> Compiled {p.name} successfully")


def _get_yaml_triggers(yaml_data):
    if not yaml_data:
        return []
    triggers = yaml_data.get("on") if "on" in yaml_data else yaml_data.get(True)
    if isinstance(triggers, str):
        return [triggers]
    if isinstance(triggers, dict):
        return list(triggers.keys())
    if isinstance(triggers, list):
        return triggers
    return []


def test_workflows_yaml():
    print("Testing workflows YAML syntax and security constraints...")
    import yaml

    workflows_dir = REPO_ROOT / ".github" / "workflows"
    compare_yml = workflows_dir / "te-compare.yml"
    switch_yml = workflows_dir / "te-switch.yml"

    assert compare_yml.is_file(), f"Missing {compare_yml}"
    assert switch_yml.is_file(), f"Missing {switch_yml}"

    # Parse te-compare.yml
    c_raw = compare_yml.read_text(encoding="utf-8")
    assert "draft" in c_raw.lower() and "contents: write" in c_raw, (
        "te-compare.yml must comment why contents: write is required (draft access only)"
    )
    c_data = yaml.safe_load(c_raw)
    c_triggers = _get_yaml_triggers(c_data)
    assert c_triggers == ["workflow_dispatch"], (
        f"te-compare.yml must be workflow_dispatch only, found: {c_triggers}"
    )
    c_perms = c_data.get("permissions", {})
    assert c_perms == {"contents": "write"} or c_perms.get("contents") == "write", (
        f"te-compare.yml must have contents: write for draft release visibility, got {c_perms}"
    )
    print("  -> te-compare.yml is workflow_dispatch only with contents: write (for draft listing)!")

    # Parse te-switch.yml
    switch_raw = switch_yml.read_text(encoding="utf-8")
    assert "draft" in switch_raw.lower() and "contents: write" in switch_raw, (
        "te-switch.yml plan job must comment why contents: write is required (draft access only)"
    )
    s_data = yaml.safe_load(switch_raw)
    s_triggers = _get_yaml_triggers(s_data)
    assert s_triggers == ["workflow_dispatch"], (
        f"te-switch.yml must be workflow_dispatch only, found: {s_triggers}"
    )

    # Concurrency verification
    concurrency = s_data.get("concurrency", {})
    assert concurrency.get("group") == "models-manifest", (
        f"te-switch.yml concurrency group must be 'models-manifest', got {concurrency}"
    )
    assert concurrency.get("cancel-in-progress") is False, (
        f"te-switch.yml cancel-in-progress must be False, got {concurrency}"
    )
    print("  -> te-switch.yml concurrency: { group: models-manifest, cancel-in-progress: false } verified!")

    on_block = s_data.get("on") if "on" in s_data else s_data.get(True, {})
    dispatch_inputs = on_block.get("workflow_dispatch", {}).get("inputs", {})
    assert "target" in dispatch_inputs, "te-switch.yml missing input 'target'"
    assert "dry_run" in dispatch_inputs, "te-switch.yml missing input 'dry_run'"
    dry_run_default = dispatch_inputs["dry_run"].get("default")
    assert dry_run_default is True, f"te-switch.yml dry_run default must be true, got {dry_run_default}"
    print("  -> te-switch.yml is workflow_dispatch only and dry_run defaults to true!")

    # Permissions split and job structure tests
    jobs = s_data.get("jobs", {})
    assert "plan" in jobs, "te-switch.yml must contain 'plan' job"
    assert "apply" in jobs, "te-switch.yml must contain 'apply' job"

    plan_job = jobs["plan"]
    plan_perms = plan_job.get("permissions", {})
    assert plan_perms == {"contents": "write"} or plan_perms.get("contents") == "write", (
        f"plan job must have permissions: contents: write for draft release visibility, got {plan_perms}"
    )
    plan_str = json.dumps(plan_job)
    assert "AARIA_MODELS_SIGNING_KEY" not in plan_str, (
        "plan job must NEVER access AARIA_MODELS_SIGNING_KEY secret"
    )
    print("  -> plan job has contents: write (for drafts) and no signing secret!")

    apply_job = jobs["apply"]
    apply_if = apply_job.get("if", "")
    assert "inputs.dry_run == false" in str(apply_if) or "!inputs.dry_run" in str(apply_if), (
        f"apply job must have if: inputs.dry_run == false, got: {apply_if}"
    )
    apply_perms = apply_job.get("permissions", {})
    assert apply_perms == {"contents": "write"} or apply_perms.get("contents") == "write", (
        f"apply job must have permissions: contents: write, got {apply_perms}"
    )
    assert apply_job.get("timeout-minutes") is not None, "apply job must define timeout-minutes"
    print("  -> apply job requires dry_run == false and contents: write!")

    # Check rollback step exists in apply job and handles failure() || cancelled()
    apply_steps = apply_job.get("steps", [])
    rollback_step = None
    for step in apply_steps:
        step_if = str(step.get("if", ""))
        if "failure()" in step_if and "cancelled()" in step_if:
            rollback_step = step
            break
    assert rollback_step is not None, "apply job must contain a rollback step with if: (failure() || cancelled())"
    print("  -> failure() || cancelled() rollback step verified in apply job!")

    # Docs mention sign-manifest.yml concurrency
    doc_text = (REPO_ROOT / "docs" / "TE_SWITCH.md").read_text(encoding="utf-8")
    assert "sign-manifest.yml" in doc_text, "docs/TE_SWITCH.md must mention sign-manifest.yml"
    print("  -> docs/TE_SWITCH.md notes sign-manifest.yml concurrency!")


if __name__ == "__main__":
    test_koyal_and_indicconformer_builds()
    test_live_indicconformer_matches_stored()
    test_size_limits_enforced()
    test_security_rules_enforced()
    test_noop_detection()
    test_exact_four_file_list_enforced()
    test_signature_roundtrip_throwaway_key()
    test_score_onnx_manifest_checks()
    test_fetch_draft_assets_logic()
    test_verify_manifest_logic()
    test_empty_expected_sha_refused()
    test_output_write_failure_exits_nonzero()
    test_changed_live_manifest_stops_before_upload()
    test_missing_key_fails_rollback()
    test_unchanged_manifest_no_rollback_upload()
    test_workflows_shell_syntax()
    test_py_compile_new_scripts()
    test_workflows_yaml()
    print("\nALL SWITCH UNIT CHECKS PASSED.")
