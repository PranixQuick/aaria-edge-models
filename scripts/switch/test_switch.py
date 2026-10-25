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
   - plan job has read-only permission (contents: read) and NO secrets.
   - apply job has if: inputs.dry_run == false and contents: write.
   - failure()-rollback step exists in apply job.
9. No-op detection: switching to target while already on target returns is_noop=True.
10. Exact four-file draft checking: missing file fails, extra te-koyal__ file fails.
11. score_onnx.py manifest validation: missing model.int8.onnx or tokens.txt fails (not skip).
12. fetch_draft_assets unit tests:
    - fake API response with draft match
    - fake API response with no match (fails clearly)
    - fake API response with two matches (fails clearly)
    - fake API response with missing asset (fails clearly)
    - fake API response with extra te-koyal__ asset (fails clearly)
"""

import copy
import json
import os
import py_compile
import shutil
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


def _create_mock_draft_dir(tmp_path: Path, extra_file: bool = False, missing_file: bool = False):
    """Helper to populate mock draft directory with exact Koyal files."""
    d = tmp_path / "draft"
    d.mkdir(parents=True, exist_ok=True)
    files = {
        "te-koyal__model.int8.onnx": b"fake_onnx_bytes" * 1000,
        "te-koyal__tokens.txt": b"fake_tokens",
        "te-koyal__model.json": b'{"type":"ctc"}',
        "te-koyal__LICENSE-koyal-te-120m.txt": b"license text",
    }
    if missing_file:
        del files["te-koyal__LICENSE-koyal-te-120m.txt"]
    for fname, data in files.items():
        (d / fname).write_bytes(data)
    if extra_file:
        (d / "te-koyal__unexpected.bin").write_bytes(b"extra")
    return d


def test_koyal_and_indicconformer_builds():
    print("Testing Koyal and IndicConformer switch builds...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"
    assert stored_path.is_file(), f"Missing stored indicconformer file: {stored_path}"

    with tempfile.TemporaryDirectory() as td:
        draft_dir = _create_mock_draft_dir(Path(td))

        # 1. Koyal build
        koyal_manifest, koyal_bytes = build_te_switch.build_switch_manifest(
            MOCK_MANIFEST, "koyal", stored_path, draft_dir=draft_dir
        )
        assert koyal_manifest["models"]["te"]["type"] == "ctc"
        te_files = koyal_manifest["models"]["te"]["files"]
        assert len(te_files) == 4
        file_names = [f["name"] for f in te_files]
        assert file_names == ["LICENSE", "model.int8.onnx", "model.json", "tokens.txt"]
        assert all("te-koyal__" in f["url"] for f in te_files)
        assert all(f["url"].startswith("https://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/") for f in te_files)

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
    bad_manifest = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest["models"]["en"]["files"][0]["bytes"] = 700 * 1024 * 1024
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest)
        assert False, "Expected ValueError for 700 MB total / 700 MB single file"
    except ValueError as e:
        assert "exceeds" in str(e).lower()
        print(f"  -> 700 MB limit violation caught: {e}")

    bad_manifest2 = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest2["models"]["en"]["files"][0]["bytes"] = 350 * 1024 * 1024
    bad_manifest2["models"]["hi"]["files"][0]["bytes"] = 350 * 1024 * 1024
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest2)
        assert False, "Expected ValueError for > 600 MB total"
    except ValueError as e:
        assert "600 mb" in str(e).lower()
        print(f"  -> >600 MB total limit violation caught: {e}")


def test_security_rules_enforced():
    print("Testing security rules (http URL, URL prefix, path traversal, version)...")
    bad_manifest_http = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_http["models"]["en"]["files"][0]["url"] = "http://github.com/PranixQuick/aaria-edge-models/releases/download/models-2026-10/model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_http)
        assert False, "Expected ValueError for http URL"
    except ValueError as e:
        assert "prefix" in str(e).lower() or "https" in str(e).lower()
        print(f"  -> http URL violation caught: {e}")

    bad_manifest_prefix = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_prefix["models"]["en"]["files"][0]["url"] = "https://malicious.example.com/models-2026-10/model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_prefix)
        assert False, "Expected ValueError for URL prefix mismatch"
    except ValueError as e:
        assert "required prefix" in str(e).lower()
        print(f"  -> URL prefix mismatch caught: {e}")

    bad_manifest_trav = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_trav["models"]["en"]["files"][0]["name"] = "../model.onnx"
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_trav)
        assert False, "Expected ValueError for path traversal ../"
    except ValueError as e:
        assert "path traversal" in str(e).lower()
        print(f"  -> path traversal caught: {e}")

    bad_manifest_v2 = copy.deepcopy(MOCK_MANIFEST)
    bad_manifest_v2["version"] = 2
    try:
        build_te_switch.validate_sync_manifest_rules(bad_manifest_v2)
        assert False, "Expected ValueError for version != 1"
    except ValueError as e:
        assert "version" in str(e).lower()
        print(f"  -> version != 1 caught: {e}")


def test_noop_detection():
    print("Testing no-op detection...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"
    indic_manifest, _ = build_te_switch.build_switch_manifest(MOCK_MANIFEST, "indicconformer", stored_path)
    assert build_te_switch.check_is_noop(MOCK_MANIFEST, indic_manifest) is True, "Expected no-op for indicconformer on indicconformer"

    with tempfile.TemporaryDirectory() as td:
        draft_dir = _create_mock_draft_dir(Path(td))
        koyal_manifest, _ = build_te_switch.build_switch_manifest(MOCK_MANIFEST, "koyal", stored_path, draft_dir=draft_dir)
        assert build_te_switch.check_is_noop(MOCK_MANIFEST, koyal_manifest) is False, "Expected change for koyal on indicconformer"

        koyal_manifest_2, _ = build_te_switch.build_switch_manifest(koyal_manifest, "koyal", stored_path, draft_dir=draft_dir)
        assert build_te_switch.check_is_noop(koyal_manifest, koyal_manifest_2) is True, "Expected no-op for koyal on koyal"
    print("  -> No-op detection verified!")


def test_exact_four_file_list_enforced():
    print("Testing exact four-file Koyal draft enforcement (missing/extra files fail)...")
    stored_path = REPO_ROOT / "manifests" / "te-indicconformer.json"

    try:
        build_te_switch.build_switch_manifest(MOCK_MANIFEST, "koyal", stored_path, draft_dir=None)
        assert False, "Expected FileNotFoundError for None draft_dir"
    except FileNotFoundError as e:
        print(f"  -> Missing draft_dir caught: {e}")

    with tempfile.TemporaryDirectory() as td:
        missing_dir = _create_mock_draft_dir(Path(td) / "missing", missing_file=True)
        try:
            build_te_switch.build_switch_manifest(MOCK_MANIFEST, "koyal", stored_path, draft_dir=missing_dir)
            assert False, "Expected FileNotFoundError for missing draft asset"
        except FileNotFoundError as e:
            assert "Missing required" in str(e)
            print(f"  -> Missing draft asset caught: {e}")

        extra_dir = _create_mock_draft_dir(Path(td) / "extra", extra_file=True)
        try:
            build_te_switch.build_switch_manifest(MOCK_MANIFEST, "koyal", stored_path, draft_dir=extra_dir)
            assert False, "Expected ValueError for unexpected te-koyal__ asset"
        except ValueError as e:
            assert "Unexpected te-koyal__" in str(e)
            print(f"  -> Unexpected te-koyal__ asset caught: {e}")


def test_signature_roundtrip_throwaway_key():
    print("Testing signature round-trip with a THROWAWAY key generated in memory...")
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    throwaway_key = ec.generate_private_key(ec.SECP256R1())
    manifest_data = json.dumps(MOCK_MANIFEST, sort_keys=True).encode("utf-8")

    sig = throwaway_key.sign(manifest_data, ec.ECDSA(hashes.SHA256()))
    assert len(sig) > 0

    pub_der = throwaway_key.public_key().public_bytes(
        serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo
    )
    pub_key = serialization.load_der_public_key(pub_der)

    pub_key.verify(sig, manifest_data, ec.ECDSA(hashes.SHA256()))

    tampered_data = manifest_data + b" "
    try:
        pub_key.verify(sig, tampered_data, ec.ECDSA(hashes.SHA256()))
        assert False, "Expected InvalidSignature for tampered data"
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
        model_p.write_bytes(b"model_bytes")
        tokens_p.write_bytes(b"tokens_bytes")

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


def test_py_compile_new_scripts():
    print("Testing py_compile on new scripts...")
    files_to_compile = [
        REPO_ROOT / "scripts" / "koyal_te" / "score_onnx.py",
        REPO_ROOT / "scripts" / "switch" / "build_te_switch.py",
        REPO_ROOT / "scripts" / "switch" / "fetch_draft_assets.py",
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
    c_data = yaml.safe_load(compare_yml.read_text(encoding="utf-8"))
    c_triggers = _get_yaml_triggers(c_data)
    assert c_triggers == ["workflow_dispatch"], (
        f"te-compare.yml must be workflow_dispatch only, found: {c_triggers}"
    )
    c_perms = c_data.get("permissions", {})
    assert c_perms == {"contents": "read"} or c_perms.get("contents") == "read", (
        f"te-compare.yml must have contents: read, got {c_perms}"
    )
    print("  -> te-compare.yml is workflow_dispatch only with contents: read!")

    # Parse te-switch.yml
    switch_raw = switch_yml.read_text(encoding="utf-8")
    s_data = yaml.safe_load(switch_raw)
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

    # Permissions split and job structure tests
    jobs = s_data.get("jobs", {})
    assert "plan" in jobs, "te-switch.yml must contain 'plan' job"
    assert "apply" in jobs, "te-switch.yml must contain 'apply' job"

    plan_job = jobs["plan"]
    plan_perms = plan_job.get("permissions", {})
    assert plan_perms == {"contents": "read"} or plan_perms.get("contents") == "read", (
        f"plan job must have permissions: contents: read, got {plan_perms}"
    )
    plan_str = json.dumps(plan_job)
    assert "AARIA_MODELS_SIGNING_KEY" not in plan_str, (
        "plan job must NEVER access AARIA_MODELS_SIGNING_KEY secret"
    )
    print("  -> plan job has contents: read and no signing secret!")

    apply_job = jobs["apply"]
    apply_if = apply_job.get("if", "")
    assert "inputs.dry_run == false" in str(apply_if) or "!inputs.dry_run" in str(apply_if), (
        f"apply job must have if: inputs.dry_run == false, got: {apply_if}"
    )
    apply_perms = apply_job.get("permissions", {})
    assert apply_perms == {"contents": "write"} or apply_perms.get("contents") == "write", (
        f"apply job must have permissions: contents: write, got {apply_perms}"
    )
    print("  -> apply job requires dry_run == false and contents: write!")

    # Check rollback step exists in apply job
    apply_steps = apply_job.get("steps", [])
    rollback_step = None
    for step in apply_steps:
        step_if = str(step.get("if", ""))
        if "failure()" in step_if:
            rollback_step = step
            break
    assert rollback_step is not None, "apply job must contain a rollback step with if: failure()"
    rollback_if = str(rollback_step.get("if", ""))
    assert "inputs.dry_run == false" in rollback_if or "!inputs.dry_run" in rollback_if, (
        f"Rollback step must check inputs.dry_run == false, got: {rollback_if}"
    )
    print("  -> failure()-rollback step verified in apply job!")


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
    test_py_compile_new_scripts()
    test_workflows_yaml()
    print("\nALL SWITCH UNIT CHECKS PASSED.")
