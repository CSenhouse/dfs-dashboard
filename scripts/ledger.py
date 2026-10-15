"""DraftKings transaction history -> weekly profit and loss.

DraftKings has no export, and its Transaction History page only keeps the
rows on screen, so the history arrives as text copied from the page in
overlapping chunks. Each transaction is eight lines:

    September 7, 2025
    8:55am
    Wagers and Entry Fees and Purchases
    Fantasy contest entry fee
    ID: C1517F0E-C51D-C69C-8660-CC7B7EBB100F
    Fantasy
    -$2.00
    $230.00

Chunks may overlap and may arrive in any order: rows are de-duplicated on
their ID, and the running balance is used to check that nothing is missing --
a gap between two copied chunks shows up as a balance that does not chain.

Keep the input in data/ledger/ (git-ignored: it is a financial record).

    PYTHONPATH=src .venv/Scripts/python.exe scripts/ledger.py
    PYTHONPATH=src .venv/Scripts/python.exe scripts/ledger.py data/ledger/*.txt --season 2026
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

LEDGER = Path(__file__).resolve().parents[1] / "data" / "ledger"

_DATE = re.compile(r"^(January|February|March|April|May|June|July|August|"
                   r"September|October|November|December) \d{1,2}, \d{4}$")
_TIME = re.compile(r"^\d{1,2}:\d{2}\s?[ap]m$", re.I)
_ID = re.compile(r"^ID:\s*([0-9A-F-]{36})$", re.I)
_MONEY = re.compile(r"^([+-]?)\$([\d,]+\.\d{2})$")


def _money(s: str) -> float | None:
    m = _MONEY.match(s.strip())
    if not m:
        return None
    v = float(m.group(2).replace(",", ""))
    return -v if m.group(1) == "-" else v


def parse(text: str) -> pd.DataFrame:
    """Every transaction found in `text`; anything else is ignored."""
    lines = [l.strip() for l in text.splitlines() if l.strip()]
    rows = []
    i = 0
    while i + 7 < len(lines):
        block = lines[i:i + 8]
        idm = _ID.match(block[4])
        amount, balance = _money(block[6]), _money(block[7])
        if (_DATE.match(block[0]) and _TIME.match(block[1]) and idm
                and amount is not None and balance is not None):
            when = pd.to_datetime(f"{block[0]} {block[1]}",
                                  format="%B %d, %Y %I:%M%p", errors="coerce")
            rows.append({"when": when, "pos": len(rows),
                         "category": block[2],
                         "description": block[3], "id": idm.group(1).upper(),
                         "product": block[5], "amount": amount,
                         "balance": balance})
            i += 8
        else:
            i += 1
    return pd.DataFrame(rows)


def kind(r) -> str:
    d = r["description"].lower()
    if "entry fee" in d:
        return "entry"
    if "refund" in d:
        return "refund"
    if "winning" in d or "payout" in d or "prize" in d:
        return "winnings"
    if r["category"].lower().startswith("promotions") or "credit" in d:
        return "promo"
    if "deposit" in d:
        return "deposit"
    if "withdraw" in d:
        return "withdrawal"
    return "other"


def check_chain(df: pd.DataFrame) -> pd.DataFrame:
    """Rows where the balance does not follow from the previous row --
    a sign that transactions are missing between them."""
    d = df.sort_values(["when", "pos"], ascending=[True, False]
                       ).reset_index(drop=True)
    expected = d["balance"].shift(1) + d["amount"]
    bad = d[(expected - d["balance"]).abs() > 0.005].iloc[1:] \
        if len(d) > 1 else d.iloc[:0]
    return bad


def nfl_week(ts: pd.Timestamp) -> str:
    """Tuesday-to-Monday weeks, labelled by the Sunday inside them."""
    sunday = (ts - pd.Timedelta(days=(ts.weekday() - 1) % 7)
              + pd.Timedelta(days=5)).normalize()
    return sunday.strftime("%Y-%m-%d")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*")
    ap.add_argument("--season", type=int, default=None,
                    help="only transactions from Aug 1 of this year onward")
    args = ap.parse_args()
    files = [Path(f) for f in args.files] or sorted(LEDGER.glob("*.txt"))
    if not files:
        print(f"no files -- save copied history to {LEDGER}\\*.txt")
        return 1
    text = "\n".join(f.read_text(encoding="utf-8-sig", errors="replace")
                     for f in files)
    raw = parse(text)
    if raw.empty:
        print("no transactions recognised")
        return 1
    # DK lists newest first, so within one minute later rows on the page are
    # earlier transactions.
    df = (raw.drop_duplicates("id")
             .sort_values(["when", "pos"], ascending=[True, False])
             .reset_index(drop=True))
    print(f"{len(raw)} rows read, {len(df)} unique transactions, "
          f"{df['when'].min():%b %d, %Y} to {df['when'].max():%b %d, %Y}")

    gaps = check_chain(df)
    if gaps.empty:
        print("balances chain correctly: nothing missing")
    else:
        print(f"WARNING: {len(gaps)} place(s) where transactions are missing "
              "(copy the chunk covering these again):")
        for r in gaps.itertuples():
            print(f"  just before {r.when:%b %d %Y %I:%M%p}")

    if args.season:
        df = df[df["when"] >= pd.Timestamp(args.season, 8, 1)]
    df["kind"] = df.apply(kind, axis=1)
    df["week"] = df["when"].map(nfl_week)
    fantasy = df[df["product"].str.lower().eq("fantasy")
                 | df["kind"].isin(["promo"])]
    table = fantasy.pivot_table(index="week", columns="kind", values="amount",
                                aggfunc="sum", fill_value=0.0)
    for c in ("entry", "refund", "winnings", "promo"):
        if c not in table.columns:
            table[c] = 0.0
    table["contest_net"] = table["entry"] + table["refund"] + table["winnings"]
    table["entries"] = fantasy[fantasy["kind"] == "entry"].groupby(
        "week")["amount"].size().reindex(table.index).fillna(0).astype(int)
    out = table[["entries", "entry", "refund", "winnings", "contest_net",
                 "promo"]]
    print("\nweek of (Sunday)  entries  fees  refunds  winnings  contests net  "
          "promos")
    for wk, r in out.iterrows():
        print(f"  {wk}    {int(r['entries']):>5}  {r['entry']:>7.2f}  "
              f"{r['refund']:>6.2f}  {r['winnings']:>8.2f}  "
              f"{r['contest_net']:>+10.2f}  {r['promo']:>+7.2f}")
    print(f"\ntotal contest net {out['contest_net'].sum():+.2f} on "
          f"{-out['entry'].sum() - out['refund'].sum():.2f} of net fees; "
          f"weeks up {int((out['contest_net'] > 0).sum())} of {len(out)}")
    other = sorted(set(df.loc[df["kind"] == "other", "description"]))
    if other:
        print("unclassified descriptions:", "; ".join(other))
    df.to_csv(LEDGER / "transactions_parsed.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
