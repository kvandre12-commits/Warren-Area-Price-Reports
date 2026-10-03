# Warren-Area Price Reports

Automated **meat & rotisserie competitor price reporting** for a Sam's Club meat
department in Warren, Ohio — benchmarked against the local big-box grocers.

Built where the butcher block meets the terminal: it turns a messy pile of
hand-collected prices into a clean, manager-ready snapshot (CSV + PDF + chart +
plain-English summary) so the meat team can see where they stand at a glance.

---

## What it does

- **Parses messy real-world prices** — handles `$5.99/lb`, `5.99 per lb`,
  `(1.20)` negatives, stray characters, and double decimals without choking.
- **Compares against competitors** — Walmart, Meijer, ALDI, and Giant Eagle in
  the Warren area, configured per store.
- **Generates a full report bundle**:
  - `report_latest.csv` — the cleaned, ranked data
  - `report.pdf` — a shareable one-pager
  - `chart.png` — a visual price comparison
  - `manager_summary.txt` — the TL;DR for a busy department lead

## Configuration

Everything is driven by [`config.yaml`](config.yaml) — the store name, the items
to track (ribeye, 80/20 ground beef, rotisserie chicken, pork butt, ...), the
competitor list, and the output filenames. No code edits required to retarget it
to a different store or basket of items.

## Usage

```bash
pip install pyyaml matplotlib reportlab
python meat_price_report/report.py
```

Prices live in `meat_price_report/prices.csv`; the report reads config + CSV and
writes the bundle described above.

## Why it exists

This is a working crossover project from someone who spends real shifts in a meat
department *and* writes code: take a repetitive, error-prone manual task
(eyeballing competitor prices) and turn it into a repeatable, auditable pipeline.
Sharp tools, clean structure, deliberate output.
