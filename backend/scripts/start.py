"""Start uvicorn with SSL certs auto-resolved to absolute paths.

The mkcert certs live at repo_root/certificates/{cert,key}.pem.
When ``uv --directory backend run ...`` is used, the CWD changes to
``backend/`` which breaks relative cert paths.  This script resolves
certs from __file__ so they always resolve correctly regardless of CWD.

Usage (via pnpm):
    pnpm backend
"""
from __future__ import annotations

import logging
from pathlib import Path

import uvicorn

HERE = Path(__file__).resolve().parent        # backend/scripts/
REPO_ROOT = HERE.parent.parent                # omni-main/

cert = REPO_ROOT / "certificates" / "cert.pem"
key  = REPO_ROOT / "certificates" / "key.pem"

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    log = logging.getLogger("start")

    if cert.exists() and key.exists():
        log.info("SSL: %s", cert.parent)
        ssl_kwargs: dict = dict(ssl_certfile=str(cert), ssl_keyfile=str(key))
    else:
        log.warning("SSL: certs not found at %s — starting in HTTP mode", cert.parent)
        ssl_kwargs = {}

    uvicorn.run(
        "omni_backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        reload_dirs=[str(REPO_ROOT / "backend" / "src")],
        **ssl_kwargs,
    )
