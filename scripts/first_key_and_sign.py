"""ONE-TIME: create the ECDSA P-256 signing key, sign release/manifest.json, and store the private key ONLY as the
GitHub Actions secret AARIA_MODELS_SIGNING_KEY of PranixQuick/aaria-edge-models (write-only: GitHub never shows it
again; only the sign-manifest workflow can use it). The key exists only in this process's memory. Prints the PUBLIC
key only. The GitHub token is taken from `git credential fill` and never printed.

Usage: python first_key_and_sign.py <release_dir>      (needs: pip install cryptography pynacl requests)
"""
import base64, json, subprocess, sys
from pathlib import Path
import requests
from nacl import encoding, public
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import ec

REPO = "PranixQuick/aaria-edge-models"
root = Path(sys.argv[1])
cred = subprocess.run(["git", "credential", "fill"], input=b"protocol=https\nhost=github.com\n\n",
                      capture_output=True, check=True).stdout.decode()
token = next(l.split("=", 1)[1] for l in cred.splitlines() if l.startswith("password="))
H = {"Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"}

key = ec.generate_private_key(ec.SECP256R1())
data = (root / "manifest.json").read_bytes()
(root / "manifest.json.sig").write_bytes(key.sign(data, ec.ECDSA(hashes.SHA256())))
pub_der = key.public_key().public_bytes(serialization.Encoding.DER, serialization.PublicFormat.SubjectPublicKeyInfo)
(root / "manifest_public_key.b64").write_text(base64.b64encode(pub_der).decode())

pem = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.PKCS8, serialization.NoEncryption())
r = requests.get(f"https://api.github.com/repos/{REPO}/actions/secrets/public-key", headers=H, timeout=30); r.raise_for_status()
k = r.json()
sealed = public.SealedBox(public.PublicKey(k["key"].encode(), encoding.Base64Encoder())).encrypt(pem)
r = requests.put(f"https://api.github.com/repos/{REPO}/actions/secrets/AARIA_MODELS_SIGNING_KEY", headers=H, timeout=30,
                 json={"encrypted_value": base64.b64encode(sealed).decode(), "key_id": k["key_id"]})
r.raise_for_status()
del pem, key, token
print("Signed manifest.json; private key stored only as GitHub Actions secret AARIA_MODELS_SIGNING_KEY.")
print("PUBLIC key (safe to share):", base64.b64encode(pub_der).decode())
