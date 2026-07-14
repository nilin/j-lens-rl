"""Finalize the V6 image while proving the cache-build token did not leak."""

import os

from modal_finalize_image import main as finalize_shared_image


def main() -> None:
    if os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN leaked beyond the isolated V6 cache build layer")
    finalize_shared_image()


if __name__ == "__main__":
    main()
