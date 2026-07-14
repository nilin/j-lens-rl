"""Fail closed unless the registered fresh tournament Volume exists as v2."""

from collections.abc import Callable
from typing import Any

import modal


VOLUME_NAME = "j-lens-rl-development-emotional-tournament-u5-h15-20260714b"


def verify_tournament_v1_volume_v2(
    volume_factory: Callable[..., Any] | None = None,
) -> str:
    factory = volume_factory or modal.Volume.from_name
    try:
        volume = factory(VOLUME_NAME, create_if_missing=False, version=2)
        volume.hydrate()
    except Exception as error:
        raise RuntimeError(
            "refusing tournament launch because its registered fresh v2 Volume "
            "placeholder does not exist"
        ) from error
    object_id = getattr(volume, "object_id", None)
    if not isinstance(object_id, str) or not object_id:
        raise RuntimeError("hydrated tournament Volume lacks a stable object identity")
    return object_id


if __name__ == "__main__":
    print(verify_tournament_v1_volume_v2())
