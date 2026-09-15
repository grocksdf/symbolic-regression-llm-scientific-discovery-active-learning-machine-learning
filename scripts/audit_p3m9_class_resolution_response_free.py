"""Audit P3M.9 H0 class resolution without opening held-out data."""
from __future__ import annotations
import argparse, csv, hashlib, json
from collections import defaultdict
from pathlib import Path

def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()

def build(root: Path) -> dict[str, object]:
    root = root.resolve(); summary = json.loads((root/'summary.json').read_text())
    manifest = json.loads((root/'RUN_MANIFEST.json').read_text())
    rows = list(csv.DictReader((root/'tables'/'per_seed_policy_metrics.csv').open(encoding='utf-8')))
    pcpi = [r for r in rows if r['policy'].startswith('pcpi_')]
    by_family = defaultdict(list)
    for r in pcpi: by_family[r['dataset_family']].append(r)
    family = {}
    for name, values in sorted(by_family.items()):
        counts = sorted({int(r['initial_operational_class_count']) for r in values})
        structures = sorted({int(round(float(r['initial_operational_class_count']) / max(1e-15, 1.0-float(r['initial_class_aggregation_fraction'])))) for r in values})
        thresholds = sorted({float(r['operational_class_distance_threshold']) for r in values})
        family[name] = {
            'registered_initial_class_counts': counts,
            'registered_structure_counts': structures,
            'registered_thresholds': thresholds,
            'aggregation_fractions': sorted({float(r['initial_class_aggregation_fraction']) for r in values}),
            'degenerate': all(float(r['initial_class_aggregation_fraction']) == 0.0 for r in values),
        }
    checks = {
        'heldout_closed': not bool(summary.get('heldout_opened', True)) and not bool(manifest.get('heldout_opened', True)),
        'protocol_complete': int(summary.get('successful_runs', -1)) == int(summary.get('expected_runs', -2)) and int(summary.get('failure_count', -1)) == 0,
        'threshold_budget_derived': manifest.get('budgets', {}).get('operational_class_resolution_method') == 'one-unit-aggregate-predictive-separation',
        'shared_threshold': len({t for v in family.values() for t in v['registered_thresholds']}) == 1,
        'gas_degeneracy_observed': bool(family.get('uci_gas_turbine', {}).get('degenerate')),
    }
    status = 'root-cause-confirmed-no-class-claim' if all(checks.values()) else 'audit-incomplete'
    return {'schema':'pcpi-p3m9-class-resolution-response-free-audit-v1','source_output':str(root),'source_summary_sha256':sha(root/'summary.json'),'source_manifest_sha256':sha(root/'RUN_MANIFEST.json'),'checks':checks,'families':family,'root_cause':'H0 decision-regret class geometry is finer than the preregistered one-unit aggregate separation at the matched budget; Gas has no initial aggregation. This is not a held-out effect.','status':status,'confirmatory_authorization':False}

def main():
    p=argparse.ArgumentParser(); p.add_argument('output',type=Path); p.add_argument('--report',type=Path); a=p.parse_args(); report=build(a.output); target=(a.report or a.output/'p3m9_class_resolution_audit.json'); target.write_text(json.dumps(report,indent=2,sort_keys=True)+'\n',encoding='utf-8'); print(json.dumps(report,indent=2,sort_keys=True)); return 0 if report['status']=='root-cause-confirmed-no-class-claim' else 2
if __name__=='__main__': raise SystemExit(main())
