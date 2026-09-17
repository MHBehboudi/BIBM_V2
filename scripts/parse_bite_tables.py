#!/usr/bin/env python3
"""Extract BiTE's PUBLISHED per-subject within-subject accuracies from the xlsx tables shipped in its
repository (third_party/BiteEEG/results/result_table/, pinned commit 924eb322) into results/published_bite_tables.json.

openpyxl is not required: an xlsx is a zip of XML. Cells are read from the first worksheet using the
shared-strings table. Rows are model names; columns are Sub1..SubN, then 'Accuracy Mean±Std',
'Kappa Mean±Std', 'p-value', 'Significance'. BiTE's own row is labelled 'Proposed'.
"""
import json, re, sys, zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "third_party/BiteEEG/results/result_table"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
FILES = {"2a": "2a - Within Subject.xlsx", "2b": "2b - Within Subject.xlsx",
         "hgd": "hgd - Within Subject.xlsx", "sdssvep": "sdssvep - Within Subject Short.xlsx"}


def col_index(ref):
    letters = re.match(r"[A-Z]+", ref).group(0)
    n = 0
    for ch in letters:
        n = n * 26 + ord(ch) - 64
    return n - 1


def read_sheet(path):
    with zipfile.ZipFile(path) as z:
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", NS):
                shared.append("".join(t.text or "" for t in si.iter(f"{{{NS['m']}}}t")))
        sheet = sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet"))[0]
        rows = []
        for row in ET.fromstring(z.read(sheet)).find("m:sheetData", NS).findall("m:row", NS):
            cells = {}
            for c in row.findall("m:c", NS):
                v = c.find("m:v", NS)
                if v is None:
                    inline = c.find("m:is", NS)
                    value = "".join(t.text or "" for t in inline.iter(f"{{{NS['m']}}}t")) if inline is not None else None
                elif c.get("t") == "s":
                    value = shared[int(v.text)]
                else:
                    value = v.text
                cells[col_index(c.get("r"))] = value
            rows.append([cells.get(i) for i in range(max(cells) + 1)] if cells else [])
        return rows


def main():
    out = {"source": "BiTE repository results/result_table/*.xlsx at commit 924eb32241ba1a7c80dbc4ba097f8c979da17578",
           "note": "seed 2025, final epoch; 'Proposed' is BiTE", "tables": {}}
    for ds, name in FILES.items():
        rows = [r for r in read_sheet(TABLES / name) if any(v not in (None, "") for v in r)]
        header = rows[0]
        subs = [i for i, h in enumerate(header) if h and re.fullmatch(r"Sub\d+", str(h))]
        acc_col = next(i for i, h in enumerate(header) if h and str(h).startswith("Accuracy"))
        table = {}
        for r in rows[1:]:
            model = r[0]
            if not model:
                continue
            per = {}
            for i in subs:
                try:
                    per[int(header[i][3:])] = float(r[i])
                except (TypeError, ValueError):
                    pass
            if not per:
                continue
            table["BiTE" if model == "Proposed" else model] = {
                "per_subject": per, "mean_reported": r[acc_col],
                "mean_of_subjects": round(sum(per.values()) / len(per), 4)}
        out["tables"][ds] = table
        print(ds, {m: v["mean_of_subjects"] for m, v in table.items()})
    target = ROOT / "results/published_bite_tables.json"
    target.write_text(json.dumps(out, indent=1))
    print("->", target)


if __name__ == "__main__":
    sys.exit(main())
