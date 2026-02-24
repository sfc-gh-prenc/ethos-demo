"""Polars pipeline for converting raw decoded tokens into structured dicts."""

import asyncio

import polars as pl
from ethos.utils import group_tokens_by_info


def format_tokens_as_dicts(
    tokens: list[str],
    decile_maps: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Sync wrapper around :func:`format_tokens_as_dicts_async`."""
    return asyncio.run(format_tokens_as_dicts_async(tokens, decile_maps))


async def format_tokens_as_dicts_async(
    tokens: list[str],
    decile_maps: dict[str, dict[str, str]] | None = None,
) -> list[dict]:
    """Convert raw decoded tokens into a list of dicts (one per clinical event).

    Uses the Polars pipeline ported from the ETHOS notebook to group related tokens, pivot
    categories into columns, and strip null values.

    When *decile_maps* is provided (output of :func:`build_decile_label_maps`), DECILE,
    SBP_DECILE and DBP_DECILE values are annotated with their numeric ranges
    (e.g. ``"6"`` becomes ``"6 [14 - 25]"``).
    """
    if not tokens:
        return []

    groups = group_tokens_by_info(tokens)
    groups = groups[: len(tokens)]

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

    cats = set(df["cat"].unique().to_list())

    lf = (
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
    )

    if decile_maps:
        exprs = _decile_annotation_exprs(decile_maps, cats)
        if exprs:
            lf = lf.with_columns(*exprs)

    df = await lf.collect_async()

    return [{k: v for k, v in d.items() if v is not None} for d in df.to_dicts()]


def _decile_annotation_exprs(
    maps: dict[str, dict[str, str]],
    cats: set[str],
) -> list[pl.Expr]:
    """Build Polars expressions to annotate decile values with range labels."""
    exprs: list[pl.Expr] = []

    if "DECILE" in cats:
        combined = {**maps.get("vital", {}), **maps.get("lab", {})}
        if combined:
            name_sources: list[pl.Expr] = []
            if "VITAL" in cats:
                name_sources.append(
                    pl.when(
                        pl.col("VITAL").is_not_null() & (pl.col("VITAL") != "BLOOD_PRESSURE")
                    ).then(pl.col("VITAL"))
                )
            if "LAB" in cats:
                name_sources.append(pl.when(pl.col("LAB").is_not_null()).then(pl.col("LAB")))
            if name_sources:
                name_col = pl.coalesce(*name_sources)
                exprs.append(
                    pl.when(name_col.is_not_null() & pl.col("DECILE").is_not_null())
                    .then(
                        pl.concat_str(name_col, pl.lit("||"), "DECILE").replace(
                            combined, default=pl.col("DECILE")
                        )
                    )
                    .otherwise(pl.col("DECILE"))
                    .alias("DECILE")
                )

    if "SBP_DECILE" in cats and maps.get("sbp"):
        exprs.append(pl.col("SBP_DECILE").replace(maps["sbp"]).alias("SBP_DECILE"))

    if "DBP_DECILE" in cats and maps.get("dbp"):
        exprs.append(pl.col("DBP_DECILE").replace(maps["dbp"]).alias("DBP_DECILE"))

    return exprs
