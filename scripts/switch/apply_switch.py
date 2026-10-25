#!/usr/bin/env python3
"""
Apply the Telugu model switch safely to release models-2026-10.

Order of operations:
1. Plan hand-off verification: validates plan artifact manifest exists and matches expected SHA-256.
   Apply NEVER rebuilds the manifest.
2. No-op verification: checks if models.te is already identical; if so, exits 0 cleanly.
3. Local signing: signs the plan manifest locally with AARIA_MODELS_SIGNING_KEY (ECDSA P-256 SHA-256).
   Never logs or prints secrets.
4. Pre-upload verification: verifies the generated signature locally with manifest_public_key.b64
   BEFORE any changes touch the live release.
5. Safe Koyal upload (if target is koyal):
   - For each of the 4 Koyal files: checks if live asset already exists on release.
   - If it exists and SHA-256 matches draft: skips upload.
   - If it exists and SHA-256 differs: FAILS before touching anything!
   - Uploads new assets without --clobber.
   - Re-downloads all 4 Koyal assets from release and verifies SHA-256 matches manifest.
6. Timestamped previous manifest upload:
   - Uploads manifest-previous-<UTC timestamp>.json and manifest-previous-<UTC timestamp>.json.sig
   - Never clobbers old ones (preserves immutable audit history).
7. Live manifest update:
   - Uploads new manifest.json.sig and manifest.json with --clobber.
8. Exact post-upload after-check:
   - Downloads live manifest.json and manifest.json.sig.
   - Asserts byte-for-byte equality with local signed manifest.
   - Verifies signature using release public key.
"""

import argparse
import base64
import datetime
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path


KOYAL_REQUIRED_ASSETS = [
    "te-koyal__model.int8.onnx",
    "te-koyal__tokens.txt",
    "te-koyal__model.json",
    "te-koyal__LICENSE-koyal-te-120m.txt",
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def upload_with_retry(release: str, files: list, clobber: bool = False, max_attempts: int = 3, delay: int = 5):
    """Upload files to a GitHub release with exponential-like retry."""
    cmd = ["gh", "release", "upload", release] + [str(f) for f in files]
    if clobber:
        cmd.append("--clobber")

    for attempt in range(1, max_attempts + 1):
        print(f"Upload attempt {attempt} of {max_attempts}: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("Upload succeeded.")
            return
        print(f"Upload attempt {attempt} failed (code {res.returncode}).")
        if attempt < max_attempts:
            time.sleep(delay)

    raise RuntimeError(f"All {max_attempts} upload attempts failed for files: {files}")


def get_live_release_assets(release: str) -> dict:
    """Return dictionary of {asset_name: asset_info} from gh release view."""
    res = subprocess.run(
        ["gh", "release", "view", release, "--json", "assets"],
        capture_output=True,
        text=True,
    )
    if res.returncode != 0:
        raise RuntimeError(f"Failed to fetch live release assets for {release}: {res.stderr}")
    data = json.loads(res.stdout)
    return {a["name"]: a for a in data.get("assets", [])}


def sign_manifest(manifest_bytes: bytes, pem_key: str) -> bytes:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    k = serialization.load_pem_private_key(pem_key.encode("utf-8"), password=None)
    return k.sign(manifest_bytes, ec.ECDSA(hashes.SHA256()))


def verify_signature(manifest_bytes: bytes, sig_bytes: bytes, pubkey_b64: str) -> None:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pub_der = base64.b64decode(pubkey_b64.strip())
    pub_key = serialization.load_der_public_key(pub_der)
    pub_key.verify(sig_bytes, manifest_bytes, ec.ECDSA(hashes.SHA256()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, choices=["koyal", "indicconformer"], help="Switch target")
    parser.add_argument("--current-dir", required=True, type=Path, help="Directory containing current manifest files")
    parser.add_argument("--plan-manifest", required=True, type=Path, help="Path to built manifest.json from plan")
    parser.add_argument("--expected-sha256", type=str, default=None, help="Expected SHA-256 of plan manifest")
    parser.add_argument("--draft-dir", type=Path, default=Path("draft_assets"), help="Directory containing draft assets")
    parser.add_argument("--release", type=str, default="models-2026-10", help="Release tag")
    args = parser.parse_args()

    print("=================================================================")
    print("APPLYING TELEGRAPHED MODEL SWITCH (LIVE UPDATE)")
    print("=================================================================")

    # 1. Plan hand-off verification: check that exact manifest built in plan is used
    if not args.plan_manifest.is_file():
        print(f"::error::Plan manifest artifact not found: {args.plan_manifest}", file=sys.stderr)
        return 1

    manifest_bytes = args.plan_manifest.read_bytes()
    manifest_sha = hashlib.sha256(manifest_bytes).hexdigest()
    print(f"Plan artifact manifest SHA-256: {manifest_sha}")

    if args.expected_sha256:
        expected = args.expected_sha256.strip().lower()
        if manifest_sha != expected:
            print(
                f"::error::Plan -> apply hand-off SHA-256 mismatch! "
                f"Plan printed {expected}, but downloaded artifact has {manifest_sha}. "
                f"Refusing to continue.",
                file=sys.stderr,
            )
            return 1
        print(f"SUCCESS: Plan hand-off SHA-256 verified ({manifest_sha}). Apply will not rebuild.")

    # Check companion .sha256 file if present
    companion_sha_file = args.plan_manifest.parent / f"{args.plan_manifest.name}.sha256"
    if companion_sha_file.is_file():
        companion_line = companion_sha_file.read_text(encoding="utf-8").strip()
        companion_hash = companion_line.split()[0].lower()
        if companion_hash != manifest_sha:
            print(
                f"::error::Companion .sha256 file mismatch: expected {companion_hash}, got {manifest_sha}",
                file=sys.stderr,
            )
            return 1
        print("SUCCESS: Companion .sha256 file matched.")

    # 2. Check no-op
    current_manifest_file = args.current_dir / "manifest.json"
    if not current_manifest_file.is_file():
        print(f"::error::Current manifest not found: {current_manifest_file}", file=sys.stderr)
        return 1

    current_data = json.loads(current_manifest_file.read_text(encoding="utf-8"))
    new_data = json.loads(manifest_bytes.decode("utf-8"))

    if current_data.get("models", {}).get("te") == new_data.get("models", {}).get("te"):
        print("No change - nothing to do")
        print("Live models.te is already identical to switch target. Exiting cleanly.")
        return 0

    # 3. Check signing key
    signing_key = os.environ.get("AARIA_MODELS_SIGNING_KEY", "").strip()
    if not signing_key:
        print("::error::AARIA_MODELS_SIGNING_KEY secret is missing. Cannot sign manifest.", file=sys.stderr)
        return 1

    # 4. Sign new manifest locally
    print("Signing new manifest.json locally with AARIA_MODELS_SIGNING_KEY...")
    try:
        sig_bytes = sign_manifest(manifest_bytes, signing_key)
    except Exception as e:
        print(f"::error::Failed to sign manifest: {e}", file=sys.stderr)
        return 1

    sig_file = args.plan_manifest.parent / "manifest.json.sig"
    sig_file.write_bytes(sig_bytes)
    print("Signed new manifest.json successfully.")

    # 5. Verify signature locally with release public key BEFORE touching release
    pubkey_file = args.current_dir / "manifest_public_key.b64"
    if not pubkey_file.is_file():
        print(f"::error::Release public key file not found: {pubkey_file}", file=sys.stderr)
        return 1

    pubkey_b64 = pubkey_file.read_text(encoding="utf-8")
    try:
        verify_signature(manifest_bytes, sig_bytes, pubkey_b64)
        print("PRE-UPLOAD LOCAL SIGNATURE VERIFIED: New signature verified with release public key.")
    except Exception as e:
        print(f"::error::Pre-upload signature verification failed: {e}", file=sys.stderr)
        return 1

    # 6. Koyal asset check and upload (Fix 5: No silent overwrite)
    if args.target == "koyal":
        print("Checking Koyal assets on live release...")
        live_assets = get_live_release_assets(args.release)

        to_upload = []
        check_dir = Path("check_existing_koyal")
        check_dir.mkdir(parents=True, exist_ok=True)

        for asset_name in KOYAL_REQUIRED_ASSETS:
            draft_file = args.draft_dir / asset_name
            if not draft_file.is_file():
                print(f"::error::Missing required draft asset in {args.draft_dir}: {asset_name}", file=sys.stderr)
                return 1

            draft_sha = sha256_file(draft_file)

            if asset_name in live_assets:
                print(f"Asset {asset_name} already exists on live release {args.release}. Downloading to compare...")
                dl_res = subprocess.run(
                    ["gh", "release", "download", args.release, "-p", asset_name, "-D", str(check_dir), "--clobber"],
                    capture_output=True,
                    text=True,
                )
                if dl_res.returncode != 0:
                    print(f"::error::Failed to download existing asset {asset_name}: {dl_res.stderr}", file=sys.stderr)
                    return 1

                live_file = check_dir / asset_name
                live_sha = sha256_file(live_file)
                if live_sha == draft_sha:
                    print(f"Asset {asset_name} on live release is byte-identical (SHA: {draft_sha}). Skipping upload.")
                else:
                    print(
                        f"::error::CRITICAL: Live asset {asset_name} exists with DIFFERENT SHA-256: "
                        f"live={live_sha} vs draft={draft_sha}. Refusing to overwrite! Aborting before touching release.",
                        file=sys.stderr,
                    )
                    return 1
            else:
                to_upload.append(draft_file)

        if to_upload:
            print(f"Uploading {len(to_upload)} new Koyal file(s) to {args.release} (no clobber)...")
            upload_with_retry(args.release, to_upload, clobber=False)
        else:
            print("All Koyal files already exist on live release with identical digests. No uploads required.")

        # Re-download all 4 Koyal files from release and check sha256 against new manifest
        print("Re-downloading Koyal files from live release to verify SHA-256 against manifest...")
        verify_assets_dir = Path("verify_assets")
        verify_assets_dir.mkdir(parents=True, exist_ok=True)
        dl_all_cmd = ["gh", "release", "download", args.release]
        for a in KOYAL_REQUIRED_ASSETS:
            dl_all_cmd.extend(["-p", a])
        dl_all_cmd.extend(["-D", str(verify_assets_dir), "--clobber"])

        res = subprocess.run(dl_all_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            print(f"::error::Failed to re-download Koyal assets for verification: {res.stderr}", file=sys.stderr)
            return 1

        te_files = new_data["models"]["te"]["files"]
        for f in te_files:
            aname = f["url"].split("/")[-1]
            p = verify_assets_dir / aname
            if not p.is_file():
                print(f"::error::Missing verified asset: {aname}", file=sys.stderr)
                return 1
            h = sha256_file(p)
            if h != f["sha256"]:
                print(f"::error::SHA mismatch on verified asset {aname}: {h} vs {f['sha256']}", file=sys.stderr)
                return 1
            print(f"Verified {aname} SHA-256 match: {h}")

    # 7. Upload timestamped previous manifest (Fix 5: never clobber old ones)
    utc_now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prev_manifest_name = f"manifest-previous-{utc_now}.json"
    prev_sig_name = f"manifest-previous-{utc_now}.json.sig"

    history_dir = Path("manifest_history")
    history_dir.mkdir(parents=True, exist_ok=True)
    ts_manifest = history_dir / prev_manifest_name
    ts_sig = history_dir / prev_sig_name

    shutil.copy2(args.current_dir / "manifest.json", ts_manifest)
    shutil.copy2(args.current_dir / "manifest.json.sig", ts_sig)

    print(f"Uploading immutable previous manifest history: {prev_manifest_name} and {prev_sig_name} (no clobber)...")
    upload_with_retry(args.release, [ts_manifest, ts_sig], clobber=False)

    # 8. Upload new manifest.json.sig and manifest.json (updating live active release)
    print("Uploading active manifest.json.sig and manifest.json to live release...")
    upload_with_retry(args.release, [sig_file, args.plan_manifest], clobber=True)

    # 9. Exact post-upload after-check: download and verify byte-for-byte equality & signature
    print("Performing exact post-upload after-check from live release...")
    post_dir = Path("post_verify")
    post_dir.mkdir(parents=True, exist_ok=True)
    dl_post = subprocess.run(
        ["gh", "release", "download", args.release, "-p", "manifest.json", "-p", "manifest.json.sig", "-D", str(post_dir), "--clobber"],
        capture_output=True,
        text=True,
    )
    if dl_post.returncode != 0:
        print(f"::error::Failed to download manifest for post-upload verification: {dl_post.stderr}", file=sys.stderr)
        return 1

    downloaded_manifest = (post_dir / "manifest.json").read_bytes()
    downloaded_sig = (post_dir / "manifest.json.sig").read_bytes()

    if downloaded_manifest != manifest_bytes:
        print("::error::EXACT AFTER-CHECK FAILED: downloaded manifest.json is not byte-for-byte equal to uploaded manifest!", file=sys.stderr)
        return 1

    if downloaded_sig != sig_bytes:
        print("::error::EXACT AFTER-CHECK FAILED: downloaded manifest.json.sig is not byte-for-byte equal to generated signature!", file=sys.stderr)
        return 1

    print("EXACT AFTER-CHECK PASSED: Live manifest.json and manifest.json.sig are byte-for-byte identical.")

    try:
        verify_signature(downloaded_manifest, downloaded_sig, pubkey_b64)
        print("POST-UPLOAD SIGNATURE VERIFIED: Downloaded manifest verified with release public key.")
    except Exception as e:
        print(f"::error::POST-UPLOAD SIGNATURE VERIFICATION FAILED: {e}", file=sys.stderr)
        return 1

    print("=================================================================")
    print("SWITCH SUCCESSFULLY EXECUTED AND VERIFIED.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
