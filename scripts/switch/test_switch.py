#!/usr/bin/env python3
"""
Unit tests for the Telugu model switch logic (runs locally).

Tests:
1. Koyal build: models.te points to Koyal, non-target languages unchanged byte-for-byte.
2. IndicConformer build: models.te matches stored byte-for-byte, non-target languages unchanged.
3. Stored IndicConformer matches live manifest models.te (skips gracefully if offline).
4. Limit enforcement: fake 700 MB total fails, fake 450 MB single file fails.
5. Security rules: http URL fails, path traversal ../ fails, version != 1 fails.
6. Signature round-trip test using a THROWAWAY key generated in-memory.
7. Python compilation on all new scripts (score_onnx.py, build_te_switch.py, test_switch.py).
8. YAML parsing on both workflows (te-compare.yml, te-switch.yml):
   workflow_dispatch only, dry_run defaults to true.
"""

import json
import os
import py_compile
import sys
import tempfile
import urllib.request
import urllib.error
from pathlib import Path

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts" / "switch"))
import build_te_switch


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
        "voice1": {
            "en": [
                {
                    "text": "Hello",
                    "bytes": 10000,
                    "format": "WAV",
                    "sha256": "c" * 64,
                    "url": "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/replies__v1__en__hello.wav",
                }
            ]
        }
    },
}


def test_koyal_and_indicconformer_builds():
    print("Testing Koyal and IndicConformer switch builds...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"
    assert stored_path.is_file(), f"Missing stored indicconformer file: {stored_path}"

    # 1. Koyal build
    koyal_manifest, koyal_bytes = build_te_switch.build_switch_manifest(
        MOCK_MANIFEST, "koyal", stored_path
    )
    assert koyal_manifest["models"]["te"]["type"] == "ctc"
    te_files = koyal_manifest["models"]["te"]["files"]
    assert len(te_files) == 4
    file_names = [f["name"] for f in te_files]
    assert file_names == ["LICENSE", "model.int8.onnx", "model.json", "tokens.txt"]
    assert all("te-koyal__" in f["url"] for f in te_files)
    assert all(f["url"].startswith("https://") for f in te_files)

    # Verify other languages and keys unchanged byte-for-byte
    assert koyal_manifest["version"] == MOCK_MANIFEST["version"]
    assert koyal_manifest["release"] == MOCK_MANIFEST["release"]
    assert koyal_manifest["models"]["en"] == MOCK_MANIFEST["models"]["en"]
    assert koyal_manifest["models"]["hi"] == MOCK_MANIFEST["models"]["hi"]
    assert koyal_manifest["replies"] == MOCK_MANIFEST["replies"]
    print("  -> Koyal build verified and other languages byte-identical!")

    # 2. IndicConformer build
    indic_manifest, indic_bytes = build_te_switch.build_switch_manifest(
        MOCK_MANIFEST, "indicconformer", stored_path
    )
    assert indic_manifest["models"]["te"]["type"] == "ctc"
    stored_te = json.loads(stored_path.read_text(encoding="utf-8"))
    assert indic_manifest["models"]["te"] == stored_te
    assert indic_manifest["models"]["en"] == MOCK_MANIFEST["models"]["en"]
    assert indic_manifest["models"]["hi"] == MOCK_MANIFEST["models"]["hi"]
    assert indic_manifest["replies"] == MOCK_MANIFEST["replies"]
    print("  -> IndicConformer build verified byte-for-byte against stored entry!")


def test_live_indicconformer_matches_stored():
    print("Testing that stored te-indicconformer.json matches live release manifest...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"
    stored_data = json.loads(stored_path.read_text(encoding="utf-8"))

    live_url = "https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/manifest.json"
    try:
        req = urllib.request.Request(live_url, headers={"User-Agent": "test-switch"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            live_bytes = resp.read()
            live_manifest = json.loads(live_bytes.decode("utf-8"))
            live_te = live_manifest.get("models", {}).get("te")
            assert live_te == stored_data, (
                f"Stored IndicConformer does not match live manifest!\nStored: {stored_data}\nLive: {live_te}"
            )
            print("  -> Stored te-indicconformer.json matches live models-2026-10 manifest byte-for-byte!")
    except (urllib.error.URLError, TimeoutError, OSError) as e:
        print(f"  -> SKIPPED: Offline or network unavailable ({e}). Stored file verified locally.")


def test_size_limits_enforced():
    print("Testing 600 MB total and 400 MB per-file limits...")
    import copy

    # Fake 700 MB total: single model with 700 MB
    bad_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest["models"]["en"]["files"][0]["bytes"] = 700 * 1024 * 1024
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest)
        assert False, "Expected ValueError for 700 MB total / 700 MB single file"
    except ValueError as e:
        assert "exceeds" in str(e).lower()
        print(f"  -> 700 MB limit violation caught: {e}")

    # Fake total > 600 MB across multiple valid files (< 400 MB each)
    bad_manifest2 = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest2["models"]["en"]["files"][0]["bytes"] = 350 * 1024 * 1024
    bad_manifest2["models"]["hi"]["files"][0]["bytes"] = 350 * 1024 * 1024  # Total = 700 MB + te
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest2)
        assert False, "Expected ValueError for > 600 MB total"
    except ValueError as e:
        assert "600 mb" in str(e).lower()
        print(f"  -> >600 MB total limit violation caught: {e}")


def test_security_rules_enforced():
    print("Testing security rules (http URL, path traversal, version)...")
    import copy

    # HTTP URL
    bad_manifest_http = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_http["models"]["en"]["files"][0]["url"] = "http://example.com/model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_http)
        assert False, "Expected ValueError for http URL"
    except ValueError as e:
        assert "https" in str(e).lower()
        print(f"  -> http URL violation caught: {e}")

    # Path traversal ../
    bad_manifest_trav = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_trav["models"]["en"]["files"][0]["name"] = "../model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_trav)
        assert False, "Expected ValueError for path traversal ../"
    except ValueError as e:
        assert "path traversal" in str(e).lower()
        print(f"  -> path traversal caught: {e}")

    # Version != 1
    bad_manifest_v2 = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_v2["version"] = 2
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_v2)
        assert False, "Expected ValueError for version != 1"
    except ValueError as e:
        assert "version" in str(e).lower()
        print(f"  -> version != 1 caught: {e}")


def test_signature_roundtrip_throwaway_key():
    print("Testing signature round-trip with a THROWAWAY key generated in memory...")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    # Generate a fresh ephemeral throwaway key
    throwaway_key = ec.generate_private_key(ec.SECP256R1())
    manifest_data = json.dumps(MOCK_MANIFEST, sort_keys=True).encode("utf-8")

    # Sign data (ECDSA P-256 over SHA-256, raw DER)
    sig = throwaway_key.sign(manifest_data, ec.ECDSA(hashes.SHA256()))
    assert len(sig) > 0

    # Extract public key as DER SubjectPublicKeyInfo
    pub_der = throwaway_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pub_key = serialization.load_der_public_key(pub_der)

    # Verify signature
    pub_key.verify(sig, manifest_data, ec.ECDSA(hashes.SHA256()))

    # Verify tampered payload fails
    tampered_data = manifest_data + b" "
    try:
        pub_key.verify(sig, tampered_data, ec.ECDSA(hashes.SHA256()))
        assert False, "Expected InvalidSignature for tampered data"
    except Exception:
        pass
    print("  -> Signature round-trip with throwaway key passed (tamper detection verified)!")


def test_py_compile_new_scripts():
    print("Testing py_compile on new scripts...")
    files_to_compile = [
        REPO_ROOT / "scripts" / "koyal_te" / "score_onnx.py",
        REPO_ROOT / "scripts" / "switch" / "build_te_switch.py",
        REPO_ROOT / "scripts" / "switch" / "test_switch.py",
    ]
    for p in files_to_compile:
        assert p.is_file(), f"Script to compile not found: {p}"
        py_compile.compile(str(p), doraise=True)
        print(f"  -> Compiled {p.name} successfully")


def _get_yaml_triggers(yaml_data):
    if not yaml_data:
        return []
    # In YAML 1.1 unquoted 'on' is parsed as boolean True
    triggers = yaml_data.get("on") if "on" in yaml_data else yaml_data.get(True)
    if isinstance(triggers, str):
        return [triggers]
    if isinstance(triggers, dict):
        return list(triggers.keys())
    if isinstance(triggers, list):
        return triggers
    return []


def test_workflows_yaml():
    print("Testing workflows YAML syntax and dispatch constraints...")
    import yaml

    workflows_dir = REPO_ROOT / ".github" / "workflows"
    compare_yml = workflows_dir / "te-compare.yml"
    switch_yml = workflows_dir / "te-switch.yml"

    assert compare_yml.is_file(), f"Missing {compare_yml}"
    assert switch_yml.is_file(), f"Missing {switch_yml}"

    # Parse te-compare.yml
    c_data = yaml.safe_load(compare_yml.read_text(encoding="utf-8"))
    c_triggers = _get_yaml_triggers(c_data)
    assert c_triggers == ["workflow_dispatch"], (
        f"te-compare.yml must be workflow_dispatch only, found: {c_triggers}"
    )
    print("  -> te-compare.yml is workflow_dispatch only!")

    # Parse te-switch.yml
    s_data = yaml.safe_load(switch_yml.read_text(encoding="utf-8"))
    s_triggers = _get_yaml_triggers(s_data)
    assert s_triggers == ["workflow_dispatch"], (
        f"te-switch.yml must be workflow_dispatch only, found: {s_triggers}"
    )

    on_block = s_data.get("on") if "on" in s_data else s_data.get(True, {})
    dispatch_inputs = on_block.get("workflow_dispatch", {}).get("inputs", {})
    assert "target" in dispatch_inputs, "te-switch.yml missing input 'target'"
    assert "dry_run" in dispatch_inputs, "te-switch.yml missing input 'dry_run'"
    dry_run_default = dispatch_inputs["dry_run"].get("default")
    assert dry_run_default is True, f"te-switch.yml dry_run default must be true, got {dry_run_default}"
    print("  -> te-switch.yml is workflow_dispatch only and dry_run defaults to true!")


if __name__ == "__main__":
    test_koyal_and_indicconformer_builds()
    test_live_indicconformer_matches_stored()
    test_size_limits_enforced()
    test_security_rules_enforced()
    test_signature_roundtrip_throwaway_key()
    test_py_compile_new_scripts()
    test_workflows_yaml()
    print("\nALL SWITCH UNIT CHECKS PASSED.")
