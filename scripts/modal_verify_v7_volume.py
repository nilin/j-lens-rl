"""Fail closed unless the frozen fresh V7 placeholder exists as Modal Volume v2."""

from collections.abc import Callable
from typing import Any

import modal


VOLUME_NAME = "j-lens-rl-confirmatory-v7-profanity-u5-20260714a"


def verify_v7_volume_v2(
    volume_factory: Callable[..., Any] | None = None,
) -> str:
    """Hydrate the noncreating V2 handle and return its stable object identity."""
    factory = volume_factory or modal.Volume.from_name
    try:
        volume = factory(VOLUME_NAME, create_if_missing=False, version=2)
        volume.hydrate()
    except Exception as error:
        raise RuntimeError(
            "refusing V7 launch because the registered fresh Volume placeholder "
            "does not yet exist as version 2"
        ) from error
    object_id = getattr(volume, "object_id", None)
    if not isinstance(object_id, str) or not object_id:
        raise RuntimeError("hydrated V7 Volume lacks a stable Modal object identity")
    return object_id


if __name__ == "__main__":
    print(verify_v7_volume_v2())
