"""Summarize Cortex development evaluation logs by labeled query category."""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean

METRIC = re.compile(r'^\s*Recall@5:\s*([0-9.eE+-]+),\s*RR:\s*([0-9.eE+-]+),\s*nDCG@5:\s*([0-9.eE+-]+)\s*$', re.M)

def read_log(path):
    raw = path.read_bytes()
    if raw.startswith((b'\xff\xfe', b'\xfe\xff')):
        return raw.decode('utf-16')
    return raw.decode('utf-8-sig')

def analyze(path, labels):
    groups = defaultdict(list)
    seen = set()
    for block in re.split(r'^Query: ', read_log(path), flags=re.M)[1:]:
        query = block.splitlines()[0].strip()
        if query not in labels or query in seen:
            raise ValueError(f'{path}: unknown or duplicate query: {query}')
        metrics = METRIC.findall(block)
        if len(metrics) != 2:
            raise ValueError(f'{path}: expected two metric rows for {query!r}, got {len(metrics)}')
        seen.add(query)
        semantic, hybrid = (tuple(map(float, row)) for row in metrics)
        groups[labels[query]['type']].append((query, semantic, hybrid))
    if seen != labels.keys():
        raise ValueError(f'{path}: matched {len(seen)} of {len(labels)} labeled queries')
    print(f'\n{path.name} — {len(seen)} queries')
    print(f'{"Category":32} {"N":>4} {"Sem R@5":>9} {"Hyb R@5":>9} {"Hyb MRR":>9} {"Hyb nDCG":>9} {"Incomplete":>10}')
    for category, rows in sorted(groups.items()):
        print(f'{category:32} {len(rows):4d} {mean(r[1][0] for r in rows):9.4f} {mean(r[2][0] for r in rows):9.4f} {mean(r[2][1] for r in rows):9.4f} {mean(r[2][2] for r in rows):9.4f} {sum(r[2][0] < 1 for r in rows):10d}')
    print('\nExample incomplete hybrid results (up to 2 per category):')
    for category, rows in sorted(groups.items()):
        for query, _, hybrid in sorted((r for r in rows if r[2][0] < 1), key=lambda r: r[2][0])[:2]:
            print(f'[{category}] Recall={hybrid[0]:.3f} | {query}')
            print('  Expected:', ', '.join(labels[query]['relevant_chunk_ids']))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('metadata', type=Path)
    parser.add_argument('logs', type=Path, nargs='+')
    args = parser.parse_args()
    entries = json.loads(args.metadata.read_text(encoding='utf-8-sig'))
    labels = {e['query'].strip(): e for e in entries}
    if len(labels) != len(entries):
        parser.error('duplicate metadata queries')
    for path in args.logs:
        analyze(path, labels)

if __name__ == '__main__':
    main()
