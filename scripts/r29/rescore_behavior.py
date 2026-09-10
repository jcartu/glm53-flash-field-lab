#!/usr/bin/env python3
"""Separate answer semantics from the preserved strict task-format scores.

This does not regenerate model output or replace the original receipts. JSON
selection never uses the expected answer. Code is rescored in the same isolated
container with module-level assignments permitted, including computed values.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re

import behavior_probe as scoring


def extract_json(text):
    fenced = re.findall(r'```json\s*\n(.*?)```', text, re.DOTALL | re.IGNORECASE)
    values = []
    if fenced:
        for block in fenced:
            values.append(json.loads(block))
        method = 'explicit-json-fence'
    else:
        decoder = json.JSONDecoder()
        cursor = 0
        while cursor < len(text):
            positions = [position for token in ('{', '[') if (position := text.find(token, cursor)) >= 0]
            if not positions:
                break
            start = min(positions)
            try:
                value, consumed = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                cursor = start + 1
                continue
            if isinstance(value, (dict, list)):
                values.append(value)
            cursor = start + consumed
        method = 'visible-json-fragment'
    if not values:
        raise ValueError('No JSON object or array in visible answer')
    if any(not scoring.equal_json(values[0], value) for value in values[1:]):
        raise ValueError('Multiple different JSON answers; no oracle-based selection allowed')
    return values[0], {'method': method, 'consistent_values': len(values)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--fixture', type=Path, required=True)
    parser.add_argument('--summary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    fixture_bytes = args.fixture.read_bytes()
    fixture = json.loads(fixture_bytes)
    tasks = {task['id']: task for task in fixture['tasks']}
    original_bytes = args.summary.read_bytes()
    original = json.loads(original_bytes)
    strict_sandbox = scoring.SANDBOX
    removal = '        ast.literal_eval(node.value)\n'
    if strict_sandbox.count(removal) != 1:
        raise RuntimeError('Unexpected strict scorer source; do not silently change rescoring semantics')
    scoring.SANDBOX = strict_sandbox.replace(removal, '')
    rows = []
    for record in original['results']:
        row = {'id': record['id'], 'kind': record['kind'], 'family': record['family'],
            'strict_task_passed': record['task_passed'], 'strict_semantic_field': record['semantic_passed'],
            'runtime_error': record['runtime_error'], 'budget_exhausted': record['budget_exhausted'],
            'semantic_assessed': False, 'semantic_correct': False}
        if record['runtime_error'] or record['budget_exhausted']:
            row['disposition'] = 'runtime-or-budget-failure'
        elif record['kind'] == 'profile':
            row.update({'semantic_assessed': True, 'semantic_correct': record['semantic_passed'],
                        'disposition': 'unchanged-profile-verifier'})
        else:
            task = tasks[record['id']]
            try:
                if record['kind'] == 'code':
                    verification = scoring.score_code(record.get('visible_answer', ''), task['cases'])
                    if verification.get('infrastructure_error'):
                        raise RuntimeError('Code-rescoring infrastructure failed')
                    row.update({'semantic_assessed': True, 'semantic_correct': verification['passed'],
                                'fraction': verification['fraction'], 'verifier': verification,
                                'disposition': 'same-hidden-tests-with-module-level-assignments-allowed'})
                else:
                    value, method = extract_json(record.get('visible_answer', ''))
                    row.update({'semantic_assessed': True, 'semantic_correct': scoring.equal_json(value, task['expected']),
                                'parsed_answer': value, 'selection': method, 'disposition': 'json-content-separate-from-format'})
            except (ValueError, TypeError) as error:
                row['disposition'] = 'unassessable-visible-answer'
                row['error'] = str(error)
        rows.append(row)
    result = {'schema': 'r29-behavior-semantic-rescore/v2',
        'original_summary': str(args.summary), 'original_summary_sha256': hashlib.sha256(original_bytes).hexdigest(),
        'fixture_sha256': hashlib.sha256(fixture_bytes).hexdigest(),
        'scorer_source_sha256': hashlib.sha256(Path(scoring.__file__).read_bytes()).hexdigest(),
        'strict_sandbox_sha256': hashlib.sha256(strict_sandbox.encode()).hexdigest(),
        'semantic_sandbox_sha256': hashlib.sha256(scoring.SANDBOX.encode()).hexdigest(),
        'code_policy': 'The literal-only check on module-level Assign/AnnAssign values is removed. Computed assignments, including calls, comprehensions and lambdas, are permitted under the existing container, import and hidden-case constraints. This does not certify expression purity.',
        'policy_clarification': 'The original registration called these constant expressions, which was narrower than its implementation. Version 2 describes the same accepted assignment forms precisely; it also uses the bounded-output scorer. Original registration and v1 receipts remain preserved.',
        'strict_task_successes': sum(row['strict_task_passed'] for row in rows),
        'semantic_correct': sum(row['semantic_correct'] for row in rows),
        'semantic_assessed': sum(row['semantic_assessed'] for row in rows),
        'attempted': len(rows), 'rows': rows,
        'scope': 'Post-hoc diagnostic scoring, declared after the published baseline exposed format/assignment rejections. Applied to every checkpoint equally, without new model requests, oracle-selected JSON fragments or edits to original strict scores. The registration wording was clarified after independent review; accepted assignment forms were not changed.'}
    scoring.save(args.output, result)
    print(json.dumps({key: result[key] for key in ['attempted', 'strict_task_successes', 'semantic_correct', 'semantic_assessed']}, indent=2))


if __name__ == '__main__':
    main()
