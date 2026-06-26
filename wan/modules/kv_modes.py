"""Pure helpers for KV-cache attention modes.

No torch / no heavy imports here so this module is cheap to import and
unit-testable on CPU. Used by wan/image2video_fast.py and generate_fast.py.
"""


def validate_kv_mode_flags(full_kv, sliding_window,
                           use_retrieval):
    """Raise ValueError if more than one mutually-exclusive KV mode is selected.

    The supported modes are mutually exclusive: at most one may be active.
    Zero active modes is allowed (default sink+recent local window).
    """
    selected = []
    if full_kv:
        selected.append("--full_kv")
    if sliding_window and sliding_window > 0:
        selected.append("--sliding_window")
    if use_retrieval:
        selected.append("--use_retrieval")
    if len(selected) > 1:
        raise ValueError(
            "KV mode flags are mutually exclusive, but multiple were set: "
            f"{selected}."
        )
