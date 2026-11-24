"""Share immutable GDN index materialization inside one model forward."""


def flatten_once(context, indices, token_count, query_start, build):
    # The upstream attention builder supplies one block-table tensor and query
    # prefix tensor per group. Never reuse results across different inputs or
    # forward contexts, including subsequent requests that recycle block IDs.
    cache = getattr(context, "_ascend_gdn_flat_indices", None)
    if cache is None:
        cache = {}
        context._ascend_gdn_flat_indices = cache
    key = (id(indices), token_count, id(query_start))
    entry = cache.get(key)
    if entry is not None and entry[0] is indices and entry[1] is query_start:
        return entry[2]
    value = build(indices, token_count, query_start)
    cache[key] = (indices, query_start, value)
    return value
