"""Audit development retrieval logs and prepare a positive-clause contrast experiment.

Uses standard library only. Does not change the source index or held-out queries.
"""
import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def load_labels(path):
    rows = json.loads(path.read_text(encoding='utf-8-sig'))
    labels = {r['query'].strip(): r for r in rows}
    if len(rows) != len(labels):
        raise ValueError('Duplicate metadata queries')
    return labels


def read_log(path):
    raw = path.read_bytes()
    return raw.decode('utf-16' if raw.startswith((b'\xff\xfe', b'\xfe\xff')) else 'utf-8-sig')


def read_rankings(path, labels):
    ranks = {}
    for block in re.split(r'^Query: ', read_log(path), flags=re.M)[1:]:
        query = block.splitlines()[0].strip()
        if query not in labels or query in ranks:
            raise ValueError(f'Unknown or duplicate query in {path}: {query}')
        row = re.search(r'^\s*Hybrid:[ \t]*(.*)$', block, flags=re.M)
        if row is None:
            raise ValueError(f'Missing Hybrid result row: {query}')
        ids = row.group(1).split()
        if len(ids) > 5 or len(set(ids)) != len(ids):
            raise ValueError(f'Invalid top-5 result list: {query}')
        ranks[query] = ids
    if ranks.keys() != labels.keys():
        raise ValueError(f'{path}: matched {len(ranks)} of {len(labels)} queries')
    return ranks


def email_id(chunk_id):
    return re.sub(r'_\d+$', '', chunk_id)


def audit(labels, ranks):
    groups = defaultdict(Counter)
    for query, row in labels.items():
        c = groups[row['type']]
        relevant = set(row['relevant_chunk_ids'])
        if not relevant:
            raise ValueError(f'Empty relevance set: {query}')
        results = ranks[query]
        found = relevant.intersection(results)
        missing = relevant - found
        emails = [email_id(i) for i in results]
        expected_emails = {email_id(i) for i in relevant}
        c['queries'] += 1
        c['complete'] += not missing
        c['partial'] += bool(found and missing)
        c['zero'] += not found
        c['duplicate_email_queries'] += len(set(emails)) != len(emails)
        c['duplicate_email_slots'] += len(emails) - len(set(emails))
        c['wrong_chunk_same_email'] += sum(email_id(i) in emails for i in missing)
        c['missing_chunks'] += len(missing)
        c['email_recall_sum'] += len(expected_emails.intersection(emails)) / len(expected_emails)
        c['chunk_recall_sum'] += len(found) / len(relevant)
    for category, c in sorted(groups.items()):
        print(f'\n{category} ({c["queries"]} queries)')
        print(f'  Complete / partial / zero: {c["complete"]} / {c["partial"]} / {c["zero"]}')
        print(f'  Mean chunk recall: {c["chunk_recall_sum"]/c["queries"]:.4f}')
        print(f'  Mean email recall: {c["email_recall_sum"]/c["queries"]:.4f}')
        print(f'  Queries with duplicate email results: {c["duplicate_email_queries"]}')
        print(f'  Redundant email result slots: {c["duplicate_email_slots"]}')
        print(f'  Missing labeled chunks from emails already retrieved: {c["wrong_chunk_same_email"]} / {c["missing_chunks"]}')


def prepare(labels, destination):
    # Only the explicit contrastive development subset is rewritten; labels
    # are copied unchanged. This is an experiment, not a general negation parser.
    changed = []
    for q, row in labels.items():
        if row['type'] != 'near_duplicate_disambiguation':
            continue
        parts = re.split(r',\s+rather than\s+', q, maxsplit=1, flags=re.I)
        if len(parts) != 2 or not parts[0].strip():
            raise ValueError(f'Unexpected contrast query: {q}')
        updated = parts[0].strip().rstrip('.?!') + '.'
        changed.append({**row, 'query': updated, 'original_query': q})
    if not changed or len({r['query'] for r in changed}) != len(changed):
        raise ValueError('Empty or duplicate rewritten contrast queries')
    destination.mkdir(parents=True, exist_ok=True)
    tsv = destination / 'contrast_dev.tsv'
    meta = destination / 'contrast_dev_metadata.json'
    if tsv.exists() or meta.exists():
        raise FileExistsError(f'Experiment files already exist in {destination}')
    tsv.write_text(''.join(r['query'] + '\t' + ','.join(r['relevant_chunk_ids']) + '\n' for r in changed), encoding='utf-8')
    meta.write_text(json.dumps(changed, indent=2) + '\n', encoding='utf-8')
    print(f'\nWrote {len(changed)} rewritten development queries to {tsv}')
    print(f'Original query mappings and unchanged labels: {meta}')


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('metadata', type=Path)
    p.add_argument('log', type=Path)
    p.add_argument('--prepare-contrast', type=Path)
    args = p.parse_args()
    labels = load_labels(args.metadata)
    audit(labels, read_rankings(args.log, labels))
    if args.prepare_contrast is not None:
        prepare(labels, args.prepare_contrast)


if __name__ == '__main__':
    main()
