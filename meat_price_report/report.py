from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml  # pip install pyyaml

# Chart + PDF
import matplotlib.pyplot as plt  # pip install matplotlib
from reportlab.lib.pagesizes import letter  # pip install reportlab
from reportlab.pdfgen import canvas
from reportlab.lib.units import inch


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_price(raw: str) -> float:
    if raw is None:
        raise ValueError("Empty price")
    s = str(raw).strip()
    if not s:
        raise ValueError("Empty price")

    s = s.lower().replace("per", "/")
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[^0-9\.\-\(\)]", "", s)

    neg = False
    if s.startswith("(") and s.endswith(")"):
        neg = True
        s = s[1:-1]

    if s.count(".") > 1:
        parts = s.split(".")
        s = parts[0] + "." + "".join(parts[1:])

    val = float(s)
    if neg:
        val = -abs(val)
    return val


def parse_week(raw: str) -> str:
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
                continue

            # Skip incomplete rows (lets you paste store blocks with blank prices safely)
            price_raw = (r.get("price") or "").strip()
            if not price_raw:
                continue

            week = parse_week(r["week"])
            store = str(r["store"]).strip()
            item_raw = str(r["item"]).strip()
            unit = str(r["unit"]).strip().lower()
            price = parse_price(price_raw)

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
    idx: Dict[Tuple[str, str, str], Row] = {}
    for r in rows:
        idx[(r.week, r.store, r.item_norm)] = r
    return idx


def money(x: float) -> str:
    sign = "-" if x < 0 else ""
    return f"{sign}${abs(x):.2f}"


def make_chart_png(out_rows: List[dict], png_path: Path, title: str) -> None:
    # Chart: Sam's vs Cheapest gap by item (positive means Sam's is higher)
    items = [r["item"] for r in out_rows]
    gaps = [r["diff_vs_cheapest"] for r in out_rows]

    plt.figure(figsize=(10, 4.5))
    plt.title(title)
    plt.bar(range(len(items)), gaps)
    plt.axhline(0)
    plt.xticks(range(len(items)), items, rotation=25, ha="right")
    plt.ylabel("Sam's price minus cheapest ($)")
    plt.tight_layout()
    plt.savefig(png_path, dpi=160)
    plt.close()


def write_pdf(
    pdf_path: Path,
    title: str,
    week: str,
    sam_store: str,
    summary_lines: List[str],
    out_rows: List[dict],
    chart_png: Optional[Path],
) -> None:
    c = canvas.Canvas(str(pdf_path), pagesize=letter)
    width, height = letter

    # Header
    y = height - 0.75 * inch
    c.setFont("Helvetica-Bold", 16)
    c.drawString(0.75 * inch, y, title)
    y -= 0.28 * inch
    c.setFont("Helvetica", 11)
    c.drawString(0.75 * inch, y, f"Week: {week}")
    y -= 0.18 * inch
    c.drawString(0.75 * inch, y, f"Sam's store: {sam_store}")
    y -= 0.35 * inch

    # Summary block
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Executive Summary")
    y -= 0.22 * inch
    c.setFont("Helvetica", 10)
    for line in summary_lines[:18]:
        c.drawString(0.85 * inch, y, line[:110])
        y -= 0.16 * inch
        if y < 3.5 * inch:
            break

    # Chart
    if chart_png and chart_png.exists():
        y -= 0.15 * inch
        c.setFont("Helvetica-Bold", 12)
        c.drawString(0.75 * inch, y, "Price Gap Chart")
        y -= 0.10 * inch
        # place chart
        img_w = 7.25 * inch
        img_h = 3.0 * inch
        c.drawImage(str(chart_png), 0.75 * inch, y - img_h, width=img_w, height=img_h, preserveAspectRatio=True, mask="auto")
        y -= img_h + 0.35 * inch

    # Table (top rows)
    c.setFont("Helvetica-Bold", 12)
    c.drawString(0.75 * inch, y, "Item Comparison (Sam's vs Cheapest)")
    y -= 0.22 * inch

    c.setFont("Helvetica-Bold", 9)
    headers = ["Item", "Unit", "Sam's", "Cheapest Store", "Cheapest", "Gap"]
    x = [0.75, 3.25, 3.75, 4.55, 6.55, 7.25]
    for i, h in enumerate(headers):
        c.drawString(x[i] * inch, y, h)
    y -= 0.16 * inch
    c.setFont("Helvetica", 9)

    for r in out_rows[:12]:
        c.drawString(x[0] * inch, y, str(r["item"])[:30])
        c.drawString(x[1] * inch, y, str(r["unit"])[:6])
        c.drawString(x[2] * inch, y, f'${r["sam_price"]:.2f}')
        c.drawString(x[3] * inch, y, str(r["cheapest_store"])[:18])
        c.drawString(x[4] * inch, y, f'${r["cheapest_price"]:.2f}')
        c.drawString(x[5] * inch, y, money(r["diff_vs_cheapest"]))
        y -= 0.16 * inch
        if y < 0.75 * inch:
            c.showPage()
            y = height - 0.75 * inch
            c.setFont("Helvetica-Bold", 12)
            c.drawString(0.75 * inch, y, "Item Comparison (continued)")
            y -= 0.22 * inch
            c.setFont("Helvetica", 9)

    c.save()


def main():
    cfg = load_config(Path("config.yaml"))

    files = cfg.get("files", {})
    input_csv = Path(files.get("input_csv", "prices.csv"))
    output_csv = Path(files.get("output_csv", "report_latest.csv"))
    summary_txt = Path(files.get("summary_txt", "manager_summary.txt"))
    output_pdf = Path(files.get("output_pdf", "report.pdf"))
    chart_png = Path(files.get("chart_png", "chart.png"))

    title = cfg.get("report", {}).get("title", "Meat & Rotisserie Competitor Snapshot")
    top_n = int(cfg.get("report", {}).get("top_n", 5))
    include_chart = bool(cfg.get("report", {}).get("include_chart", True))

    sam_default = cfg.get("store", {}).get("sam_name", "Sam's Club Warren")

    rows, item_names = load_rows(input_csv)
    w_latest, w_prev = latest_two_weeks(rows)
    if not w_latest:
        raise SystemExit("No usable rows found in prices.csv (check blanks).")

    idx = build_index(rows)

    stores_latest = sorted({r.store for r in rows if r.week == w_latest})
    items_latest = sorted({r.item_norm for r in rows if r.week == w_latest})

    # Sam's detection
    if sam_default in stores_latest:
        sam_store = sam_default
    else:
        sam_store = next((s for s in stores_latest if "sam" in s.lower()), stores_latest[0])

    # Optional filtering by config (items/competitors)
    cfg_items = [norm_item(x) for x in (cfg.get("items") or [])]
    if cfg_items:
        items_latest = [it for it in items_latest if it in set(cfg_items)]

    cfg_competitors = set(cfg.get("competitors") or [])
    if cfg_competitors:
        # keep sam_store + listed competitors if present
        allowed = set(cfg_competitors) | {sam_store}
        stores_latest = [s for s in stores_latest if s in allowed]

    out_rows: List[dict] = []
    wins: List[Tuple[str, float]] = []
    losses: List[Tuple[str, float]] = []
    missing: List[Tuple[str, str]] = []

    for item in items_latest:
        prices: List[Tuple[str, float]] = []
        unit = None
        for store in stores_latest:
            r = idx.get((w_latest, store, item))
            if r:
                prices.append((store, r.price))
                unit = r.unit

        if not prices:
            continue

        cheapest_store, cheapest_price = min(prices, key=lambda x: x[1])

        sam_row = idx.get((w_latest, sam_store, item))
        if not sam_row:
            missing.append((item, "Sam's missing latest week"))
            continue

        diff_vs_cheapest = sam_row.price - cheapest_price
        tag = "WIN" if diff_vs_cheapest <= 0 else "LOSS"

        sam_prev = idx.get((w_prev, sam_store, item)) if w_prev else None
        wow = (sam_row.price - sam_prev.price) if sam_prev else None

        out_rows.append({
            "week": w_latest,
            "item": display_item(item, item_names),
            "unit": unit or "",
            "sam_price": sam_row.price,
            "cheapest_store": cheapest_store,
            "cheapest_price": cheapest_price,
            "diff_vs_cheapest": diff_vs_cheapest,
            "sam_wow_change": wow,
        })

        (wins if tag == "WIN" else losses).append((item, diff_vs_cheapest))

    out_rows.sort(key=lambda r: (r["diff_vs_cheapest"], r["item"]))

    # Save CSV
    with output_csv.open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["week", "item", "unit", "sam_price", "cheapest_store", "cheapest_price", "diff_vs_cheapest", "sam_wow_change"]
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
                "sam_wow_change": (f'{r["sam_wow_change"]:+.2f}' if r["sam_wow_change"] is not None else "")
            })

    # Manager summary
    def top_n_items(items: List[Tuple[str, float]], n: int, reverse: bool) -> List[Tuple[str, float]]:
        return sorted(items, key=lambda t: t[1], reverse=reverse)[:n]

    top_losses = top_n_items(losses, top_n, reverse=True)
    top_wins = top_n_items(wins, top_n, reverse=False)

    summary: List[str] = []
    summary.append(f"{title} — Week {w_latest}")
    summary.append(f"Sam's store detected: {sam_store}")
    summary.append("")
    summary.append(f"WINS (Sam's cheapest or tied): {len(wins)} items")
    for item, diff in top_wins:
        summary.append(f"  • {display_item(item, item_names)}: {money(diff)} vs cheapest")
    summary.append("")
    summary.append(f"LOSSES (Sam's higher than cheapest): {len(losses)} items")
    for item, diff in top_losses:
        summary.append(f"  • {display_item(item, item_names)}: {money(diff)} above cheapest")
    summary.append("")
    if w_prev:
        summary.append(f"Trend note: add/maintain prior week rows to monitor week-over-week changes (previous week detected: {w_prev}).")
    else:
        summary.append("Trend note: add a previous week set to unlock week-over-week changes.")
    if missing:
        summary.append("")
        summary.append("Missing data:")
        for item, msg in missing[:10]:
            summary.append(f"  • {display_item(item, item_names)}: {msg}")

    summary_txt.write_text("\n".join(summary), encoding="utf-8")

    # Chart + PDF
    if include_chart:
        make_chart_png(out_rows, chart_png, f"{title} — Week {w_latest}")

    write_pdf(
        pdf_path=output_pdf,
        title=title,
        week=w_latest,
        sam_store=sam_store,
        summary_lines=summary,
        out_rows=out_rows,
        chart_png=(chart_png if include_chart else None),
    )

    print("\n".join(summary))
    print("\nSaved:")
    print(f"  - {output_csv.resolve()}")
    print(f"  - {summary_txt.resolve()}")
    print(f"  - {output_pdf.resolve()}")


if __name__ == "__main__":
    main()
