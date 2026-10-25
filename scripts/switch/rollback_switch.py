#!/usr/bin/env python3
"""
Roll back to previous manifest on failure or cancellation.

Restores the saved previous manifest.json and manifest.json.sig from current-dir,
then downloads them from the release and requires:
1. Byte-for-byte equality of downloaded manifest with saved previous copy.
2. Byte-for-byte equality of downloaded signature with saved previous copy.
3. Cryptographically valid signature against release public key.
"""

import argparse
import base64
import os
import subprocess
import sys
import time
from pathlib import Path


def upload_with_retry(release: str, files: list, clobber: bool = True, max_attempts: int = 3, delay: int = 5):
    """Upload files to GitHub release with retry."""
    cmd = ["gh", "release", "upload", release] + [str(f) for f in files]
    if clobber:
        cmd.append("--clobber")

    for attempt in range(1, max_attempts + 1):
        print(f"Rollback upload attempt {attempt} of {max_attempts}: {' '.join(cmd)}")
        res = subprocess.run(cmd)
        if res.returncode == 0:
            print("Rollback upload succeeded.")
            return
        print(f"Rollback upload attempt {attempt} failed (code {res.returncode}).")
        if attempt < max_attempts:
            time.sleep(delay)

    raise RuntimeError(f"All {max_attempts} rollback upload attempts failed for files: {files}")


def verify_signature(manifest_bytes: bytes, sig_bytes: bytes, pubkey_b64: str) -> None:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pub_der = base64.b64decode(pubkey_b64.strip())
    pub_key = serialization.load_der_public_key(pub_der)
    pub_key.verify(sig_bytes, manifest_bytes, ec.ECDSA(hashes.SHA256()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current-dir", required=True, type=Path, help="Directory containing saved previous manifest files")
    parser.add_argument("--release", type=str, default="models-2026-10", help="Release tag")
    args = parser.parse_args()

    print("=================================================================")
    print("EXECUTING AUTOMATIC ROLLBACK TO PREVIOUS MANIFEST")
    print("=================================================================")

    saved_manifest_path = args.current_dir / "manifest.json"
    saved_sig_path = args.current_dir / "manifest.json.sig"
    pubkey_path = args.current_dir / "manifest_public_key.b64"

    if not saved_manifest_path.is_file() or not saved_sig_path.is_file():
        print(
            f"::error::Cannot rollback: saved previous manifest files not found in {args.current_dir}. "
            f"Expected {saved_manifest_path} and {saved_sig_path}",
            file=sys.stderr,
        )
        return 1

    saved_manifest_bytes = saved_manifest_path.read_bytes()
    saved_sig_bytes = saved_sig_path.read_bytes()

    print(f"Re-uploading previous manifest.json and manifest.json.sig to {args.release}...")
    try:
        upload_with_retry(args.release, [saved_manifest_path, saved_sig_path], clobber=True)
    except Exception as e:
        print(f"::error::Failed to upload rollback manifest: {e}", file=sys.stderr)
        return 1

    print("Downloading restored manifest from release for verification...")
    rollback_verify_dir = Path("rollback_verify")
    rollback_verify_dir.mkdir(parents=True, exist_ok=True)

    dl = subprocess.run(
        ["gh", "release", "download", args.release, "-p", "manifest.json", "-p", "manifest.json.sig", "-D", str(rollback_verify_dir), "--clobber"],
        capture_output=True,
        text=True,
    )
    if dl.returncode != 0:
        print(f"::error::Failed to download restored manifest for verification: {dl.stderr}", file=sys.stderr)
        return 1

    restored_manifest_bytes = (rollback_verify_dir / "manifest.json").read_bytes()
    restored_sig_bytes = (rollback_verify_dir / "manifest.json.sig").read_bytes()

    # Byte-for-byte equality requirement
    if restored_manifest_bytes != saved_manifest_bytes:
        print(
            "::error::ROLLBACK VERIFICATION FAILED: Restored manifest.json is not byte-for-byte equal to saved copy!",
            file=sys.stderr,
        )
        return 1

    if restored_sig_bytes != saved_sig_bytes:
        print(
            "::error::ROLLBACK VERIFICATION FAILED: Restored manifest.json.sig is not byte-for-byte equal to saved copy!",
            file=sys.stderr,
        )
        return 1

    print("SUCCESS: Restored manifest.json and manifest.json.sig are byte-for-byte identical to saved previous files.")

    # Cryptographic verification
    if pubkey_path.is_file():
        pubkey_b64 = pubkey_path.read_text(encoding="utf-8")
        try:
            verify_signature(restored_manifest_bytes, restored_sig_bytes, pubkey_b64)
            print("SUCCESS: Rollback manifest signature verified with release public key.")
        except Exception as e:
            print(f"::error::Rollback signature verification failed: {e}", file=sys.stderr)
            return 1
    else:
        print(f"::warning::Public key {pubkey_path} not found; byte equality verified but signature check skipped.")

    print("=================================================================")
    print("ROLLED BACK to previous manifest successfully.")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
