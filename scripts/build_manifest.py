"""Build and sign manifest.json for PranixQuick/aaria-edge-models.

Usage:
  local (no key):  python build_manifest.py <release_dir> <release_tag> --unsigned
  in the sign-manifest GitHub workflow (key from the repo secret): python build_manifest.py <release_dir> <release_tag>

- <release_dir> holds one sub-folder per language model (en/, hi/, te/), each with model.json plus its files,
  and optionally replies/<voiceId>/<lang>/*.wav|*.ogg with a texts.json {filename: text}.
- Writes manifest.json and manifest.json.sig (DER ECDSA P-256 over SHA-256) into <release_dir>.
- Never prints the private key. If AARIA_MODELS_SIGNING_KEY is missing it stops with an error.
"""
import base64, hashlib, json, os, sys
from pathlib import Path
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

BASE = "https://github.com/PranixQuick/aaria-edge-models/releases/download"

def sha256(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def main(release_dir, tag):
    root = Path(release_dir)
    models, replies = {}, {}
    for lang_dir in sorted(p for p in root.iterdir() if p.is_dir() and p.name in ("en", "hi", "te", "kws")):
        cfg = json.loads((lang_dir / "model.json").read_text(encoding="utf-8"))
        files = []
        for f in sorted(lang_dir.iterdir()):
            if f.is_file():
                asset = f"{lang_dir.name}__{f.name}"   # release assets are flat: prefix with language
                files.append({"name": f.name, "url": f"{BASE}/{tag}/{asset}", "bytes": f.stat().st_size, "sha256": sha256(f)})
        models[lang_dir.name] = {"type": cfg["type"], "files": files}
    rdir = root / "replies"
    if rdir.is_dir():
        for vdir in sorted(p for p in rdir.iterdir() if p.is_dir()):
            for ldir in sorted(p for p in vdir.iterdir() if p.is_dir()):
                texts = json.loads((ldir / "texts.json").read_text(encoding="utf-8"))
                items = []
                for fname, text in sorted(texts.items()):
                    f = ldir / fname
                    asset = f"replies__{vdir.name}__{ldir.name}__{fname}"
                    items.append({"text": text, "url": f"{BASE}/{tag}/{asset}", "bytes": f.stat().st_size,
                                  "sha256": sha256(f), "format": "OGG" if fname.endswith(".ogg") else "WAV"})
                replies.setdefault(vdir.name, {})[ldir.name] = items
    manifest = {"version": 1, "release": tag, "models": models, "replies": replies}
    data = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    (root / "manifest.json").write_bytes(data)

    if "--unsigned" in sys.argv:
        print(f"manifest.json written ({len(data)} bytes), NOT signed (sign with first_key_and_sign.py or the sign-manifest workflow)")
        return
    pem = os.environ.get("AARIA_MODELS_SIGNING_KEY")
    if not pem:
        sys.exit("AARIA_MODELS_SIGNING_KEY not set (only the sign-manifest GitHub workflow has it). Use --unsigned locally.")
    key = serialization.load_pem_private_key(pem.encode(), password=None)
    sig = key.sign(data, ec.ECDSA(hashes.SHA256()))
    (root / "manifest.json.sig").write_bytes(sig)
    pub = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
    (root / "manifest_public_key.b64").write_text(base64.b64encode(pub).decode())
    print(f"manifest.json written ({len(data)} bytes), signed; public key saved to manifest_public_key.b64")

if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
