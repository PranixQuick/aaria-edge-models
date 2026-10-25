#!/usr/bin/env python3
"""
Verify manifest.json signature, SHA-256 digest, and byte-for-byte equality.

Used in workflows:
1. In plan to verify current manifest signature:
     verify_manifest.py --manifest current_rel/manifest.json --sig current_rel/manifest.json.sig --pubkey current_rel/manifest_public_key.b64
2. In apply to verify hand-off sha256:
     verify_manifest.py --manifest plan_artifact/new_rel/manifest.json --expected-sha256 <hash>
3. In post-apply or rollback to verify byte equality and signature:
     verify_manifest.py --manifest post_verify/manifest.json --sig post_verify/manifest.json.sig --pubkey current_rel/manifest_public_key.b64 --compare-manifest-bytes new_rel/manifest.json --compare-sig-bytes new_rel/manifest.json.sig
"""

import argparse
import base64
import hashlib
import sys
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def verify_signature(manifest_bytes: bytes, sig_bytes: bytes, pubkey_b64: str) -> None:
    from cryptography.hazmat.primitives import hashes, serialization
    from cryptography.hazmat.primitives.asymmetric import ec

    pub_der = base64.b64decode(pubkey_b64.strip())
    pub_key = serialization.load_der_public_key(pub_der)
    pub_key.verify(sig_bytes, manifest_bytes, ec.ECDSA(hashes.SHA256()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, type=Path, help="Path to manifest.json")
    parser.add_argument("--sig", type=Path, default=None, help="Path to manifest.json.sig")
    parser.add_argument("--pubkey", type=Path, default=None, help="Path to manifest_public_key.b64")
    parser.add_argument("--expected-sha256", type=str, default=None, help="Expected SHA-256 hex digest of manifest")
    parser.add_argument(
        "--compare-manifest-bytes",
        type=Path,
        default=None,
        help="Path to manifest file that must match byte-for-byte",
    )
    parser.add_argument(
        "--compare-sig-bytes",
        type=Path,
        default=None,
        help="Path to signature file that must match byte-for-byte",
    )
    args = parser.parse_args()

    if not args.manifest.is_file():
        print(f"::error::Manifest file not found: {args.manifest}", file=sys.stderr)
        return 1

    manifest_bytes = args.manifest.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)

    # 1. Check expected SHA-256
    if args.expected_sha256:
        expected = args.expected_sha256.strip().lower()
        if manifest_sha != expected:
            print(
                f"::error::Manifest SHA-256 mismatch for {args.manifest}: "
                f"expected {expected}, got {manifest_sha}",
                file=sys.stderr,
            )
            return 1
        print(f"SUCCESS: Manifest SHA-256 matches expected digest: {manifest_sha}")

    # 2. Check byte equality of manifest
    if args.compare_manifest_bytes:
        if not args.compare_manifest_bytes.is_file():
            print(
                f"::error::Comparison manifest not found: {args.compare_manifest_bytes}",
                file=sys.stderr,
            )
            return 1
        expected_bytes = args.compare_manifest_bytes.read_bytes()
        if manifest_bytes != expected_bytes:
            print(
                f"::error::Manifest byte mismatch: {args.manifest} ({len(manifest_bytes)} bytes) "
                f"differs from {args.compare_manifest_bytes} ({len(expected_bytes)} bytes)",
                file=sys.stderr,
            )
            return 1
        print(f"SUCCESS: Manifest bytes match {args.compare_manifest_bytes} byte-for-byte.")

    # 3. Check signature and pubkey
    if args.sig and args.pubkey:
        if not args.sig.is_file():
            print(f"::error::Signature file not found: {args.sig}", file=sys.stderr)
            return 1
        if not args.pubkey.is_file():
            print(f"::error::Public key file not found: {args.pubkey}", file=sys.stderr)
            return 1

        sig_bytes = args.sig.read_bytes()
        pubkey_b64 = args.pubkey.read_text(encoding="utf-8")

        try:
            verify_signature(manifest_bytes, sig_bytes, pubkey_b64)
            print(f"SUCCESS: Signature {args.sig} verified with public key {args.pubkey}.")
        except Exception as e:
            print(f"::error::Signature verification failed: {e}", file=sys.stderr)
            return 1

        # 4. Check byte equality of signature
        if args.compare_sig_bytes:
            if not args.compare_sig_bytes.is_file():
                print(
                    f"::error::Comparison signature not found: {args.compare_sig_bytes}",
                    file=sys.stderr,
                )
                return 1
            expected_sig = args.compare_sig_bytes.read_bytes()
            if sig_bytes != expected_sig:
                print(
                    f"::error::Signature byte mismatch: {args.sig} differs from {args.compare_sig_bytes}",
                    file=sys.stderr,
                )
                return 1
            print(f"SUCCESS: Signature bytes match {args.compare_sig_bytes} byte-for-byte.")
    elif args.sig or args.pubkey:
        print("::error::Both --sig and --pubkey must be provided for signature verification.", file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":
    sys.exit(main())
