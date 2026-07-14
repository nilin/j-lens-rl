"""Fail closed unless the pre-created V6 Volume C exists as Modal Volume v2."""

from collections.abc import Callable
from typing import Any

import modal


VOLUME_C_NAME = "j-lens-rl-confirmatory-v6-celebration-taper-20260714c"


def verify_volume_c_v2(
    volume_factory: Callable[..., Any] | None = None,
) -> str:
    factory = volume_factory or modal.Volume.from_name
    try:
        volume = factory(VOLUME_C_NAME, create_if_missing=False, version=2)
        volume.hydrate()
    except Exception as error:
        raise RuntimeError(
            "refusing V6 launch because fresh Volume C is absent or not version 2"
        ) from error
    object_id = getattr(volume, "object_id", None)
    if not isinstance(object_id, str) or not object_id:
        raise RuntimeError("hydrated V6 Volume C lacks a stable Modal object identity")
    return object_id


if __name__ == "__main__":
    print(verify_volume_c_v2())
