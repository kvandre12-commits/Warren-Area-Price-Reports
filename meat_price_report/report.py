from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Dict, List, Tuple, Optional


SAM_STORE_DEFAULT = "Sam's Club Warren"


def parse_price(raw: str) -> float:
    """
    Accepts: $4.99, 4.99, 4.99/lb, (4.99), -4.99, etc.
    Returns float.
    """
    if raw is None:
        raise ValueError("Empty price")
    s = str(raw).strip()
    if not s:
        raise ValueError("Empty price")

    # Remove unit suffixes like /lb, per lb, etc.
    s = s.lower()
    s = s.replace("per", "/")
    s = re.sub(r"\s+", "", s)

    # Keep digits, dot, minus, parentheses
    s = re.sub(r"[^0-9\.\-\(\)]", "", s)

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    if s.count(".") > 1:
        # e.g. "1.234.56" -> keep first two parts safely
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])

    val = float(s)
    if neg:
        val = -abs(val)
    return val


def parse_week(raw: str) -> str:
    """
    Expect YYYY-MM-DD. We keep as string but validate shape lightly.
    """
    s = str(raw).strip()
    if not re.match(r"^\d{4}-\d{2}-\d{2}$", s):
        raise ValueError(f"Bad week format: {raw} (expected YYYY-MM-DD)")
    return s


def norm_item(item: str) -> str:
    return re.sub(r"\s+", " ", str(item).strip()).lower()


def display_item(item_norm: str, original_map: Dict[str, str]) -> str:
    return original_map.get(item_norm, item_norm)


@dataclass(frozen=True)
class Row:
    week: str
    store: str
    item_norm: str
    item_raw: str
    unit: str
    price: float


def load_rows(csv_path: Path) -> Tuple[List[Row], Dict[str, str]]:
    rows: List[Row] = []
    original_item_map: Dict[str, str] = {}

    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        required = {"week", "store", "item", "unit", "price"}
        if not reader.fieldnames or not required.issubset(set(h.strip() for h in reader.fieldnames)):
            raise ValueError(f"CSV must have columns: {sorted(required)}")

        for i, r in enumerate(reader, start=2):
            if not any((r or {}).values()):
                continue  # skip empty line

            week = parse_week(r["week"])
            store = str(r["store"]).strip()
            item_raw = str(r["item"]).strip()
            unit = str(r["unit"]).strip().lower()
            price = parse_price(r["price"])

            if not store or not item_raw or not unit:
                raise ValueError(f"Missing store/item/unit on line {i}")

            item_n = norm_item(item_raw)
            original_item_map.setdefault(item_n, item_raw)

            rows.append(Row(week=week, store=store, item_norm=item_n, item_raw=item_raw, unit=unit, price=price))

    return rows, original_item_map


def latest_two_weeks(rows: List[Row]) -> Tuple[Optional[str], Optional[str]]:
    weeks = sorted({r.week for r in rows})
    if not weeks:
        return None, None
    if len(weeks) == 1:
        return weeks[-1], None
    return weeks[-1], weeks[-2]


def build_index(rows: List[Row]) -> Dict[Tuple[str, str, str], Row]:
    # key: (week, store, item_norm)
    idx: Dict[Tuple[str, str, str], Row] = {}
    for r in rows:
        idx[(r.week, r.store, r.item_norm)] = r
    return idx


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):.2f}"


def pct(diff: float, base: float) -> str:
    if base == 0:
        return "n/a"
    return f"{(diff / base) * 100:+.1f}%"


def main():
    csv_path = Path("prices.csv")
    if not csv_path.exists():
        raise SystemExit("Missing prices.csv in the current folder.")

    rows, item_names = load_rows(csv_path)
    w_latest, w_prev = latest_two_weeks(rows)
    if not w_latest:
        raise SystemExit("No rows found in prices.csv")

    idx = build_index(rows)

    # Collect latest-week universe
    stores_latest = sorted({r.store for r in rows if r.week == w_latest})
    items_latest = sorted({r.item_norm for r in rows if r.week == w_latest})

    # Pick Sam's store
    sam_store = None
    # prefer exact match, else first store containing "sam"
    if SAM_STORE_DEFAULT in stores_latest:
        sam_store = SAM_STORE_DEFAULT
    else:
        for s in stores_latest:
            if "sam" in s.lower():
                sam_store = s
                break
    if not sam_store:
        sam_store = stores_latest[0]  # fallback

    # Build analysis table for latest week
    out_rows = []
    wins = []
    losses = []
    missing = []

    for item in items_latest:
        # Find all prices for this item in latest week
        prices = []
        unit = None
        for store in stores_latest:
            r = idx.get((w_latest, store, item))
            if r:
                prices.append((store, r.price))
                unit = r.unit

        if not prices:
            continue

        # Determine cheapest
        cheapest_store, cheapest_price = min(prices, key=lambda x: x[1])

        sam_row = idx.get((w_latest, sam_store, item))
        if not sam_row:
            missing.append((item, "Sam's missing latest week"))
            continue

        diff_vs_cheapest = sam_row.price - cheapest_price  # positive means Sam's higher
        tag = "WIN" if diff_vs_cheapest <= 0 else "LOSS"

        # Week-over-week for Sam's
        sam_prev = idx.get((w_prev, sam_store, item)) if w_prev else None
        wow = None
        if sam_prev:
            wow = sam_row.price - sam_prev.price

        out_rows.append({
            "week": w_latest,
            "item": display_item(item, item_names),
            "unit": unit or "",
            "sam_price": sam_row.price,
            "cheapest_store": cheapest_store,
            "cheapest_price": cheapest_price,
            "diff_vs_cheapest": diff_vs_cheapest,
            "diff_pct_vs_cheapest": (diff_vs_cheapest / cheapest_price * 100) if cheapest_price else None,
            "sam_wow_change": wow
        })

        if tag == "WIN":
            wins.append((item, diff_vs_cheapest))
        else:
            losses.append((item, diff_vs_cheapest))

    # Sort helpful
    out_rows.sort(key=lambda r: (r["diff_vs_cheapest"], r["item"]))

    # Write outputs
    out_csv = Path("report_latest.csv")
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "week", "item", "unit",
            "sam_price",
            "cheapest_store", "cheapest_price",
            "diff_vs_cheapest", "diff_pct_vs_cheapest",
            "sam_wow_change"
        ]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in out_rows:
            w.writerow({
                "week": r["week"],
                "item": r["item"],
                "unit": r["unit"],
                "sam_price": f'{r["sam_price"]:.2f}',
                "cheapest_store": r["cheapest_store"],
                "cheapest_price": f'{r["cheapest_price"]:.2f}',
                "diff_vs_cheapest": f'{r["diff_vs_cheapest"]:.2f}',
                "diff_pct_vs_cheapest": (f'{r["diff_pct_vs_cheapest"]:+.1f}%' if r["diff_pct_vs_cheapest"] is not None else ""),
                "sam_wow_change": (f'{r["sam_wow_change"]:+.2f}' if r["sam_wow_change"] is not None else "")
            })

    # Create a manager-friendly summary
    def top_n(items: List[Tuple[str, float]], n: int, reverse: bool) -> List[Tuple[str, float]]:
        return sorted(items, key=lambda t: t[1], reverse=reverse)[:n]

    top_losses = top_n(losses, 5, reverse=True)   # biggest $ premium vs cheapest
    top_wins = top_n(wins, 5, reverse=False)      # most negative (best advantage)

    summary = []
    summary.append(f"Meat & Rotisserie Competitor Snapshot — Week {w_latest}")
    summary.append(f"Sam's store detected: {sam_store}")
    summary.append("")
    summary.append(f"WINS (Sam's is cheapest or tied): {len(wins)} items")
    for item, diff in top_wins:
        # diff negative = cheaper than cheapest? actually cheapest already; tie/cheaper means <= 0
        summary.append(f"  • {display_item(item, item_names)}: {money(diff)} vs cheapest (good)")
    summary.append("")
    summary.append(f"LOSSES (Sam's higher than cheapest): {len(losses)} items")
    for item, diff in top_losses:
        summary.append(f"  • {display_item(item, item_names)}: {money(diff)} above cheapest (risk)")
    if w_prev:
        summary.append("")
        summary.append(f"Week-over-week note: add last week data to see trends (you already have: {w_prev}).")
    else:
        summary.append("")
        summary.append("Tip: add a previous week row set to unlock week-over-week changes.")

    if missing:
        summary.append("")
        summary.append("Missing data:")
        for item, msg in missing[:10]:
            summary.append(f"  • {display_item(item, item_names)}: {msg}")

    out_txt = Path("manager_summary.txt")
    out_txt.write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("\nSaved:")
    print(f"  - {out_csv.resolve()}")
    print(f"  - {out_txt.resolve()}")


if __name__ == "__main__":
    main()
