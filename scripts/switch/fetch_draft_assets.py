#!/usr/bin/env python3
"""
Fetch assets from a GitHub release (including DRAFT releases) by tag name and asset list.

Uses GitHub REST API (GET /repos/{owner}/{repo}/releases?per_page=100) to find the release
by tag_name (drafts allowed). Then downloads each requested asset by ID via
GET /repos/{owner}/{repo}/releases/assets/{asset_id} with Accept: application/octet-stream.

Enforces:
- Tag matching: exactly one release must match the given tag
- Asset validation: all requested assets must exist
- Security: fails if any unexpected te-koyal__ assets are present on the release
- Token safety: reads GITHUB_TOKEN / GH_TOKEN from env, never prints or logs it
- Output reporting: prints each downloaded asset name, size in bytes, and hex SHA-256
"""

import argparse
import hashlib
import json
import os
import sys
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path


class SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Handles 302 redirects from GitHub API to S3/CDN without forwarding Auth header."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        new_headers = dict(req.headers)
        orig_host = urllib.parse.urlparse(req.full_url).netloc
        new_host = urllib.parse.urlparse(newurl).netloc
        if orig_host != new_host:
            new_headers.pop("Authorization", None)
            new_headers.pop("authorization", None)
        return urllib.request.Request(
            newurl,
            headers=new_headers,
            origin_req_host=req.origin_req_host,
            unverifiable=True,
        )


def sha256_file(path: Path) -> str:
    """Compute hex sha256 of file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_matching_release(releases: list, tag: str) -> dict:
    """Find exactly one release (draft allowed) matching tag_name."""
    matches = [r for r in releases if r.get("tag_name") == tag]
    if not matches:
        raise ValueError(
            f"No release found with tag_name '{tag}' (inspected {len(releases)} releases)"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Ambiguous tag: found {len(matches)} releases with tag_name '{tag}'"
        )
    return matches[0]


def validate_release_assets(release: dict, wanted_asset_names: list) -> dict:
    """
    Validate that release contains all wanted assets and no unexpected te-koyal__ assets.
    Returns mapping of asset_name -> asset_dict.
    """
    assets = release.get("assets", [])
    asset_map = {a.get("name"): a for a in assets if "name" in a}
    wanted_set = set(wanted_asset_names)

    missing = sorted([name for name in wanted_asset_names if name not in asset_map])
    if missing:
        raise ValueError(f"Missing required asset(s) in release: {missing}")

    unexpected = sorted([
        name for name in asset_map
        if name.startswith("te-koyal__") and name not in wanted_set
    ])
    if unexpected:
        raise ValueError(f"Unexpected te-koyal__ asset(s) in release: {unexpected}")

    return {name: asset_map[name] for name in wanted_asset_names}


def fetch_releases_list(repo: str, token: str) -> list:
    """Fetch releases from GitHub API (up to 100 per page)."""
    url = f"https://api.github.com/repos/{repo}/releases?per_page=100"
    headers = {
        "Accept": "application/vnd.github+json",
        "User-Agent": "aaria-edge-fetch-draft",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            data = resp.read()
            return json.loads(data.decode("utf-8"))
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error fetching releases ({e.code}): {body}") from e


def download_single_asset(repo: str, asset_id: int, token: str, out_path: Path):
    """Download single release asset by ID following redirects safely."""
    url = f"https://api.github.com/repos/{repo}/releases/assets/{asset_id}"
    headers = {
        "Accept": "application/octet-stream",
        "User-Agent": "aaria-edge-fetch-draft",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    if token:
        headers["Authorization"] = f"Bearer {token}"

    opener = urllib.request.build_opener(SafeRedirectHandler())
    req = urllib.request.Request(url, headers=headers)
    try:
        with opener.open(req) as resp:
            data = resp.read()
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(data)
    except urllib.error.HTTPError as e:
        body = e.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"GitHub API error downloading asset {asset_id} ({e.code}): {body}") from e


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="Tag name of the release (e.g. models-2026-10-koyal-te)")
    parser.add_argument("--assets", required=True, nargs="+", help="Exact list of asset names to download")
    parser.add_argument("--out-dir", required=True, type=Path, help="Directory to save downloaded assets")
    parser.add_argument(
        "--repo",
        default=os.environ.get("GITHUB_REPOSITORY", "PranixQuick/aaria-edge-models"),
        help="GitHub repository in owner/repo format",
    )
    args = parser.parse_args()

    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("::error::GITHUB_TOKEN (or GH_TOKEN) environment variable is required to access releases.", file=sys.stderr)
        sys.exit(1)

    print(f"Fetching releases for repo: {args.repo} ...")
    releases = fetch_releases_list(args.repo, token)

    try:
        rel = find_matching_release(releases, args.tag)
    except ValueError as e:
        print(f"::error::Release search failed: {e}", file=sys.stderr)
        sys.exit(1)

    rel_id = rel.get("id")
    is_draft = rel.get("draft", False)
    print(f"Found release: id={rel_id}, tag={args.tag}, draft={is_draft}")

    try:
        matched_assets = validate_release_assets(rel, args.assets)
    except ValueError as e:
        print(f"::error::Asset validation failed: {e}", file=sys.stderr)
        sys.exit(1)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {len(matched_assets)} assets to {args.out_dir} ...")

    for asset_name in args.assets:
        asset_info = matched_assets[asset_name]
        asset_id = asset_info["id"]
        target_path = args.out_dir / asset_name

        download_single_asset(args.repo, asset_id, token, target_path)

        size = target_path.stat().st_size
        sha = sha256_file(target_path)
        print(f"Downloaded {asset_name}: {size:,} bytes | SHA-256: {sha}")

    print("All draft assets successfully downloaded and verified.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
