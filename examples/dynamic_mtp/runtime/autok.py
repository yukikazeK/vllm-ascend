# SPDX-License-Identifier: Apache-2.0
"""Select a draft depth and bound it by each request's state reservation."""

from __future__ import annotations

import json
import logging
import math
import os
from pathlib import Path
from statistics import median

logger = logging.getLogger(__name__)
policy_path = Path(os.environ["DYN_FULL_POLICY"])
policy_table = {
    int(length): {int(pressure): depth for pressure, depth in row.items()}
    for length, row in json.loads(policy_path.read_text()).items()
}
reservations: dict[str, int] = {}
last_choice: tuple[int, int] | None = None


def pick(length, pressure):
    # Use the closest measured length by ratio and round up the local pressure.
    nearest = min(policy_table, key=lambda point: abs(math.log(max(1, length) / point)))
    row = policy_table[nearest]
    ceiling = next((point for point in sorted(row) if pressure <= point), max(row))
    return row[ceiling]


def prepare(scheduler):
    for rid in reservations.keys() - scheduler.requests.keys():
        del reservations[rid]
    pressure = max(1, len(scheduler.running) + len(scheduler.waiting))
    for rid, request in scheduler.requests.items():
        if rid not in reservations:
            # Keep one scratch slot for the verified width-one downgrade path.
            reservations[rid] = max(1, pick(request.num_prompt_tokens, pressure))
    if scheduler.running:
        prompts = (r.num_prompt_tokens for r in (*scheduler.running, *scheduler.waiting))
        if pick(median(prompts), pressure) == 0:
            for request in scheduler.running:
                if request.spec_token_ids:
                    request.spec_token_ids = []


def choose(scheduler, output):
    global last_choice
    scheduled = output.num_scheduled_tokens
    if not scheduled:
        return
    length = median(scheduler.requests[rid].num_prompt_tokens for rid in scheduled)
    pressure = max(len(scheduled), len(scheduler.running)) + len(scheduler.waiting)
    desired = pick(length, pressure)
    actual = min(desired, *(reservations[rid] for rid in scheduled))
    for rid, drafts in output.scheduled_spec_decode_tokens.items():
        if len(drafts) > reservations[rid]:
            raise RuntimeError("MTP draft exceeds the request's state reservation")
    output.num_spec_tokens_to_schedule = actual
    choice = (desired, actual)
    if choice != last_choice:
        logger.info("Dynamic MTP selected K=%d (desired=%d, pressure=%d, prompt=%g)",
                    actual, desired, pressure, length)
        last_choice = choice


def physical_depth(rid, maximum):
    reserved = reservations.get(rid)
    if reserved is None or reserved > maximum:
        raise RuntimeError(f"Missing or invalid MTP state reservation for {rid}")
    return reserved


def record_allocation(manager, req, count):
    """Keep the allocation callback available for diagnostic overlays."""
