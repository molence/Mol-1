# Copyright 2026 Molence Bio
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Parse and validate explicit user-supplied inference constraints."""
from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import json


def read_json(path):
    def unique_object(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"invalid JSON constant: {value}")

    return json.loads(Path(path).read_text(encoding="utf-8"),
                      object_pairs_hook=unique_object, parse_constant=invalid_constant)


def prepare_inputs(inputs, *, config):
    """Compile inline constraints into per-task roles without changing native fields."""
    tasks, rows = normalize_inputs(inputs)
    for task, row in zip(tasks, rows):
        for kind in ('site', 'exclusion'):
            if row[kind]:
                row[kind]['target_id'] = task['name']
    validate_enabled(rows, config)
    return tasks, rows


def _residue(value):
    if (not isinstance(value, list) or len(value) != 2
            or not isinstance(value[0], str) or not value[0].strip()
            or type(value[1]) is not int or value[1] < 1):
        raise ValueError('residue ID requires [chain, positive 1-based label_seq_id]')
    return tuple(value)


def _chains(value, label):
    if (not isinstance(value, list) or not value
            or any(not isinstance(c, str) or not c.strip() for c in value)
            or len(set(value)) != len(value)):
        raise ValueError(f'{label} requires unique nonempty chain strings')
    return value


def _unary(value, kind):
    if value is None:
        return None
    keys = {'source_chains', 'partner_chains', 'residue_ids'}
    if not isinstance(value, dict) or set(value) - keys:
        raise ValueError(f'{kind} requires source_chains, partner_chains and residue_ids')
    residues = value.get('residue_ids', [])
    if not isinstance(residues, list):
        raise ValueError(f'{kind}.residue_ids must be a list')
    if not residues:
        # A nonempty object with no residue field is likely a malformed request.
        if value and 'residue_ids' not in value:
            raise ValueError(f'{kind}.residue_ids is missing; use [] for none')
        return None
    source = _chains(value.get('source_chains'), f'{kind}.source_chains')
    partner = _chains(value.get('partner_chains'), f'{kind}.partner_chains')
    if set(source) & set(partner):
        raise ValueError(f'{kind}: source and partner overlap')
    ids = sorted({_residue(r) for r in residues})
    if any(chain not in source for chain, _ in ids):
        raise ValueError(f'{kind}: residue outside source chains')
    result = deepcopy(value)
    result['residue_ids'] = [list(residue) for residue in ids]
    return result


def _contact(value):
    if value is None:
        return None
    if not isinstance(value, dict) or set(value) - {'pairs', 'token_pairs'}:
        raise ValueError('contact requires pairs or token_pairs')
    if 'pairs' in value and 'token_pairs' in value:
        raise ValueError('contact cannot mix pairs and token_pairs')
    key = 'token_pairs' if 'token_pairs' in value else 'pairs'
    pairs = value.get(key, [])
    if not isinstance(pairs, list):
        raise ValueError(f'contact.{key} must be a list')
    normalized = set()
    for pair in pairs:
        if not isinstance(pair, list) or len(pair) != 2:
            raise ValueError('contact pair needs two endpoints')
        if key == 'token_pairs':
            if any(type(i) is not int or i < 0 for i in pair):
                raise ValueError('contact token IDs must be zero-based nonnegative integers')
            endpoints = pair
        else:
            endpoints = [_residue(r) for r in pair]
        if endpoints[0] == endpoints[1]:
            raise ValueError('contact endpoints must differ')
        normalized.add(tuple(sorted(endpoints)))
    return sorted(normalized) if normalized else None


def normalize_inputs(inputs):
    """Preserve native tasks and return independent per-task primitive descriptors."""
    if not isinstance(inputs, list) or not inputs:
        raise ValueError('input JSON must be a nonempty native Mol-1 task list')
    tasks, constraints, names = [], [], set()
    for original in inputs:
        if not isinstance(original, dict):
            raise ValueError('each input task must be an object')
        name = original.get('name')
        if (not isinstance(name, str) or not name.strip() or name in {'.', '..'}
                or '/' in name or '\\' in name or name in names):
            raise ValueError('task name must be unique and safe as a directory name')
        names.add(name)
        if 'constraint' in original or 'custom_site_context' in original:
            raise ValueError('only inline constraints are supported; remove constraint/custom_site_context')
        task = deepcopy(original)
        value = task.pop('constraints', None)
        if value is None:
            value = {}
        if not isinstance(value, dict) or set(value) - {'site', 'contact', 'exclusion'}:
            raise ValueError('constraints must contain only site, contact and exclusion')
        constraints.append({'site': _unary(value.get('site'), 'site'),
                            'contact': _contact(value.get('contact')),
                            'exclusion': _unary(value.get('exclusion'), 'exclusion')})
        row = constraints[-1]
        if row['site'] and row['exclusion']:
            shared_partners = set(row['site']['partner_chains']) & set(row['exclusion']['partner_chains'])
            shared_residues = (set(map(tuple, row['site']['residue_ids']))
                               & set(map(tuple, row['exclusion']['residue_ids'])))
            if shared_partners and shared_residues:
                raise ValueError('SITE and EXCLUSION conflict on shared residues and partners')
        tasks.append(task)
    return tasks, constraints


def validate_enabled(constraints, config):
    """Reject nonempty primitives the deployment cannot interpret; empty is none."""
    for row in constraints:
        for kind in ('contact', 'exclusion'):
            if row[kind] and not config.get(f'constraint_{kind}_enable', False):
                raise ValueError(f'{kind.upper()} requires constraint_{kind}_enable=true and matching checkpoint')
