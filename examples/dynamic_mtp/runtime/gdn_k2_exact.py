"""Experimental specialization for the exact-width-three GDN replay."""
from contextlib import contextmanager
from contextvars import ContextVar

EXACT_K2_ROWS = 32
WIDTH = 3
_capture_rows = ContextVar("gdn_exact_k2_capture_rows", default=0)


@contextmanager
def capture_scope(rows):
    saved = _capture_rows.set(rows)
    try:
        yield
    finally:
        _capture_rows.reset(saved)


def is_eligible(graph_rows, graph_tokens, actual_rows, actual_tokens, width, max_width):
    return (width == max_width == WIDTH and actual_rows == graph_rows
            and actual_tokens == graph_tokens == actual_rows * WIDTH)


def flatten_if_exact(state_indices, total_tokens, query_start_loc):
    rows = _capture_rows.get()
    if not rows:
        return None
    if (state_indices.ndim != 2 or query_start_loc is None
            or state_indices.shape[0] != rows or state_indices.shape[1] < WIDTH
            or total_tokens != rows * WIDTH or query_start_loc.shape[0] != rows + 1):
        raise RuntimeError("Unexpected GDN index geometry for the exact K2 graph")
    return state_indices[:, :WIDTH].contiguous().view(-1)
