"""Polars pipeline for converting raw decoded tokens into structured dicts."""

import asyncio

import polars as pl
from ethos.utils import group_tokens_by_info


def format_tokens_as_dicts(tokens: list[str]) -> list[dict]:
    """Sync wrapper around :func:`format_tokens_as_dicts_async`."""
    return asyncio.run(format_tokens_as_dicts_async(tokens))


async def format_tokens_as_dicts_async(tokens: list[str]) -> list[dict]:
    """Convert raw decoded tokens into a list of dicts (one per clinical event).

    Uses the Polars pipeline ported from the ETHOS notebook to group related tokens, pivot
    categories into columns, and strip null values.
    """
    if not tokens:
        return []

    groups = group_tokens_by_info(tokens)

    df = await (
        pl.LazyFrame([groups, pl.Series("token", tokens)])
        .with_columns(
            token=pl.when(pl.col("token").str.starts_with("BMI//"))
            .then(pl.concat_list(pl.lit("BMI"), pl.col("token").str.slice(len("BMI//"))))
            .otherwise(pl.concat_list("token"))
        )
        .explode("token")
        .select(
            "groups",
            cat=pl.when(pl.col("token").str.starts_with("HOSPITAL//"))
            .then(pl.col("token").str.splitn("//", 3).struct[1])
            .otherwise(pl.col("token").str.splitn("//", 2).struct[0])
            .replace("QUANTILE", "DECILE"),
            token=pl.when(pl.col("token").str.starts_with("ICD"))
            .then(pl.col("token").str.split("//").list.last())
            .when(pl.col("token").str.starts_with("LAB//NAME//"))
            .then(pl.col("token").str.slice(len("LAB//NAME//")).str.splitn("//", 1).struct[0])
            .otherwise(pl.col("token").str.split("//").list.last()),
        )
        .with_columns(
            cat=pl.when(pl.col("cat") == pl.col("token")).then(pl.lit("EVENT")).otherwise("cat")
        )
        .group_by("groups", maintain_order=True)
        .agg(
            "cat",
            pl.when(pl.col("cat").is_in(["ATC", "ICD_CM"]))
            .then(pl.col("token").str.join(""))
            .otherwise(pl.col("token")),
        )
        .with_columns(
            cat=pl.when(pl.col("token").list[0] == "BLOOD_PRESSURE")
            .then(["VITAL", "SBP_DECILE", "DBP_DECILE"])
            .otherwise("cat")
        )
        .explode("cat", "token")
        .with_row_index("rid")
        .collect_async()
    )

    df = await (
        df.pivot(
            index=["rid", "groups"],
            on="cat",
            values="token",
            aggregate_function="first",
        )
        .lazy()
        .drop("rid")
        .group_by("groups", maintain_order=True)
        .agg(pl.exclude("groups").drop_nulls().first())
        .drop("groups")
        .collect_async()
    )

    return [{k: v for k, v in d.items() if v is not None} for d in df.to_dicts()]
