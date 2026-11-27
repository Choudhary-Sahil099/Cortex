"""Development-only two-intent retrieval experiment for Cortex.

Subquery text is derived solely from the original query. Gold labels are used
only to satisfy the evaluate TSV format and to score the merged result lists.
No production source, held-out queries, or persistent index is modified.
"""
import argparse
import json
import math
import re
from pathlib import Path
from statistics import mean

K = 5
PATTERN = re.compile(r'Find both the manager-approved plan and the completed-work confirmation for (.+)\.', re.I)


def load_metadata(path):
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    labels = {r['query'].strip(): set(r['relevant_chunk_ids']) for r in rows}
    if len(labels) != len(rows) or any(not ids for ids in labels.values()):
        raise ValueError('Duplicate queries or missing relevance labels')
    groups = []
    for row in rows:
        if row['type'] != 'multi_relevant':
            continue
        q = row['query'].strip()
        match = PATTERN.fullmatch(q)
        if not match or len(labels[q]) != 2:
            raise ValueError(f'Unexpected compound development query: {q}')
        scope = match.group(1)
        groups.append((q, f'Find the manager-approved plan for {scope}.',
                       f'Find the completed-work confirmation for {scope}.'))
    subqueries = [s for _, a, b in groups for s in (a, b)]
    if not groups or len(set(subqueries)) != len(subqueries):
        raise ValueError('No groups or duplicate subqueries')
    return labels, groups


def prepare(labels, groups, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8', newline='\n') as f:
        for original, a, b in groups:
            # Both labels are kept for each row; individual aggregate metrics
            # are not used because each subquery is meant to address one intent.
            gold = ','.join(sorted(labels[original]))
            f.write(f'{a}\t{gold}\n{b}\t{gold}\n')
    print(f'Wrote {len(groups)*2} subqueries for {len(groups)} compound queries: {output}')


def read_rankings(path, expected):
    raw = path.read_bytes()
    text = raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')
    result = {}
    for block in re.split(r'^Query: ', text, flags=re.M)[1:]:
        q = block.splitlines()[0].strip()
        if q not in expected or q in result:
            raise ValueError(f'{path}: unexpected or duplicate query: {q}')
        pair = []
        for mode in ('Semantic-only', 'Hybrid'):
            match = re.findall(r'^\s*'+mode+r':[ \t]*([^\r\n]*)', block, flags=re.M)
            if len(match) != 1:
                raise ValueError(f'{path}: missing/duplicate {mode} ranking: {q}')
            ids = match[0].split()
            if len(ids) > K or len(set(ids)) != len(ids):
                raise ValueError(f'{path}: invalid top-{K} ranking: {q}')
            pair.append(ids)
        result[q] = tuple(pair)
    if set(result) != set(expected):
        raise ValueError(f'{path}: matched {len(result)} of {len(expected)} queries')
    return result


def merge(a, b):
    # Fixed first-intent-first round robin; no access to relevance labels.
    out = []
    seen = set()
    for i in range(max(len(a), len(b))):
        for source in (a, b):
            if i < len(source) and source[i] not in seen:
                seen.add(source[i])
                out.append(source[i])
                if len(out) == K:
                    return out
    return out


def metrics(rank, gold):
    hits = [item in gold for item in rank[:K]]
    recall = sum(hits)/len(gold)
    rr = next((1/(i+1) for i, hit in enumerate(hits) if hit), 0.0)
    dcg = sum(hit/math.log2(i+2) for i, hit in enumerate(hits))
    ideal = sum(1/math.log2(i+2) for i in range(min(K,len(gold))))
    return recall, rr, dcg/ideal


def score(labels, groups, baseline_path, sub_path):
    base = read_rankings(baseline_path, labels)
    sub = read_rankings(sub_path, {s for _, a,b in groups for s in (a,b)})
    results = {}
    for q, a,b in groups:
        gold = labels[q]
        methods = {
            'Original semantic': base[q][0],
            'Original hybrid': base[q][1],
            'Split semantic': merge(sub[a][0], sub[b][0]),
            'Split hybrid': merge(sub[a][1], sub[b][1]),
        }
        for name, rank in methods.items():
            results.setdefault(name, []).append(metrics(rank,gold))
    print(f'Compound-query evaluation: {len(groups)} original queries; final budget={K}')
    print(f'{"Method":22} {"Recall@5":>9} {"MRR":>9} {"nDCG@5":>9} {"Complete":>9} {"Partial":>8} {"Zero":>6}')
    for name, rows in results.items():
        r,m,n = (mean(x[i] for x in rows) for i in range(3))
        print(f'{name:22} {r:9.4f} {m:9.4f} {n:9.4f} {sum(x[0]==1 for x in rows):9d} {sum(0<x[0]<1 for x in rows):8d} {sum(x[0]==0 for x in rows):6d}')
    print('\nCandidate coverage (before truncation; up to 10 unique chunks):')
    for i,name in enumerate(('Semantic', 'Hybrid')):
        coverage = [len(labels[q] & (set(sub[a][i]) | set(sub[b][i])))/len(labels[q]) for q,a,b in groups]
        print(f'  {name}: mean recall={mean(coverage):.4f}; both relevant={sum(x==1 for x in coverage)}/{len(groups)}')
    print('\nNo timing claims: decomposition uses two retrieval calls per original query.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    p.add_argument('metadata', type=Path)
    p.add_argument('output', type=Path)
    p = commands.add_parser('score')
    p.add_argument('metadata', type=Path)
    p.add_argument('baseline_log', type=Path)
    p.add_argument('subquery_log', type=Path)
    args = parser.parse_args()
    labels, groups = load_metadata(args.metadata)
    if args.command == 'prepare':
        prepare(labels, groups, args.output)
    else:
        score(labels, groups, args.baseline_log, args.subquery_log)


if __name__ == '__main__':
    main()
