from __future__ import annotations


WINDOWS_DEVICE_NAMES = {"CON", "PRN", "AUX", "NUL"} | {
    f"{prefix}{number}" for prefix in ("COM", "LPT") for number in range(1, 10)
}


def is_portable_component(component: str, *, max_utf8_bytes: int = 255) -> bool:
    """Return whether one path component is portable across supported filesystems."""
    try:
        encoded_length = len(component.encode("utf-8"))
    except UnicodeEncodeError:
        return False
    if (
        not component
        or component in {".", ".."}
        or component.endswith((" ", "."))
        or encoded_length > max_utf8_bytes
        or any(ord(character) < 32 for character in component)
        or any(character in '<>:"/\\|?*' for character in component)
    ):
        return False
    device_stem = component.rstrip(" .").split(".", 1)[0].upper()
    return device_stem not in WINDOWS_DEVICE_NAMES
