#!/usr/bin/env python3
"""Regenerate compact public result tables from saved summary files."""
from __future__ import annotations
import csv, json, argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SUM = ROOT / 'analysis' / 'summaries'

def load_json(path):
    with open(path) as f: return json.load(f)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--output_dir', type=Path, default=ROOT/'analysis'/'reproduced')
    ap.add_argument('--allow_existing_output', action='store_true')
    args=ap.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.allow_existing_output:
        raise SystemExit(f'Refusing to overwrite non-empty {args.output_dir}; pass --allow_existing_output')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows=[]
    for p in sorted(SUM.rglob('summary*.json')):
        try: d=load_json(p)
        except Exception: continue
        if not isinstance(d, dict): continue
        cond=d.get('condition') or p.parent.name
        overall=d.get('overall',{})
        by=d.get('by_dimension',{})
        dim2=(by.get('dim2_word_scrambling') or by.get('2') or {}).get('accuracy')
        if 'accuracy' in overall or dim2 is not None:
            rows.append({'source':str(p.relative_to(ROOT)),'condition':cond,'overall_accuracy':overall.get('accuracy'),'dim2_accuracy':dim2})
    for p in sorted(SUM.rglob('results_summary.csv')):
        with open(p) as f:
            for r in csv.DictReader(f):
                rows.append({'source':str(p.relative_to(ROOT)),'condition':r.get('condition'),'overall_accuracy':r.get('overall_accuracy') or r.get('accuracy'),'dim2_accuracy':r.get('dim2_accuracy')})
    out=args.output_dir/'reproduced_key_results.csv'
    with out.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=['source','condition','overall_accuracy','dim2_accuracy']); w.writeheader(); w.writerows(rows)
    print(f'Wrote {out}')
if __name__=='__main__': main()
