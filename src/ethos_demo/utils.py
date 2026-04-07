"""Utility helpers for the ETHOS Demo application."""

import math

from ethos.utils import group_tokens_by_info


def wilson_margin(k: int, n: int, z: float = 1.96) -> float:
    """Half-width of the Wilson score 95% CI for a binomial proportion."""
    if n == 0:
        return 0.0
    p = k / n
    denom = 1 + z * z / n
    return z / denom * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))


def token_dict_mapping(tokens: list[str], dicts: list[dict], limit: int = 100) -> list[str]:
    """Pair each parsed dict with the source tokens that produced it.

    Returns a list of ``"<dict> <- tok1, tok2, …"`` strings (last *limit* events).
    """
    grp_ids = group_tokens_by_info(tokens)
    by_group: dict[int, list[str]] = {}
    for tok, gid in zip(tokens, grp_ids, strict=False):
        by_group.setdefault(gid, []).append(tok)
    ordered = [by_group[gid] for gid in sorted(by_group)]
    return [
        f"{d} <- {', '.join(tg)}" for d, tg in zip(dicts[-limit:], ordered[-limit:], strict=False)
    ]
