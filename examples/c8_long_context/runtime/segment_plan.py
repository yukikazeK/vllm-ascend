# SPDX-License-Identifier: Apache-2.0
"""Page-aligned segmentation for exact causal attention continuation."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Segment:
    row: int
    start: int
    length: int


@dataclass(frozen=True)
class SegmentRound:
    causal: bool
    segments: tuple


def split_history(qlens, klens, segment_tokens, block_tokens):
    if segment_tokens <= 0 or segment_tokens % block_tokens:
        raise ValueError("Segment length must be a positive whole number of cache pages")
    if len(qlens) != len(klens) or not qlens:
        raise ValueError("Expected paired nonempty query and KV length lists")
    if any(not 0 < q <= kv for q, kv in zip(qlens, klens)):
        raise ValueError("Each query must fit within its KV context")

    # Split only history that precedes every query position. Each earlier segment
    # is visible in its entirety. The remaining suffix includes all current
    # query positions and may also include a historical tail.
    # In the service, a short request stays a single causal segment even
    # when it shares a batch with a longer segmented continuation.
    prefix_counts = [(kv - q) // segment_tokens if kv - q > segment_tokens else 0
                     for q, kv in zip(qlens, klens)]
    final = tuple(Segment(row, count * segment_tokens, kv - count * segment_tokens)
                  for row, (count, kv) in enumerate(zip(prefix_counts, klens)))
    rounds = [SegmentRound(True, final)]
    for index in range(max(prefix_counts)):
        segments = tuple(Segment(row, index * segment_tokens, segment_tokens)
                         for row, count in enumerate(prefix_counts) if index < count)
        rounds.append(SegmentRound(False, segments))

    # Validate the partition on the host, independent of the NPU kernel.
    for row, (q, kv) in enumerate(zip(qlens, klens)):
        parts = sorted((s for group in rounds for s in group.segments if s.row == row),
                       key=lambda s: s.start)
        end = 0
        for part in parts:
            assert part.start == end and part.start % block_tokens == 0 and part.length > 0
            end += part.length
        assert end == kv
        assert final[row].start <= kv - q and final[row].length >= q
        assert all(part.length == segment_tokens for part in parts[:-1])
    return rounds
