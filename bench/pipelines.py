from typing import List

from pathlib import Path
import json
import numpy as np
import time
from .dataclasses import Problem, SearchResult
from .searchers.base import BaseSearcher
from scipy.stats import kendalltau


def mean_absolute_percentage_error(y_true, y_pred):
    """Small sklearn-free MAPE used by the benchmark evaluator.

    Windows application-control policies can block sklearn binary extensions.
    The evaluator only needs this simple metric, so keep it pure NumPy.
    """
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    n = min(len(y_true), len(y_pred))
    if n <= 0:
        return float("inf")
    y_true = y_true[:n]
    y_pred = y_pred[:n]
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    if not np.any(mask):
        return float("inf")
    denom = np.maximum(np.abs(y_true[mask]), 1e-12)
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / denom)))


def _failure_metrics(reason: str):
    return {
        "mse": float("inf"),
        "nmse": 100.0,
        "raw_nmse": float("inf"),
        "r2": float("-inf"),
        "kdt": float("nan"),
        "mape": float("inf"),
        "acc_0_1": 0.0,
        "restart_nmse": 100.0,
        "restart_acc_0_1": 0.0,
        "restart_best95_nmse": 100.0,
        "restart_best95_acc_0_1": 0.0,
        "max_relative_error": float("inf"),
        "raw_max_relative_error": float("inf"),
        "strict_max_relative_error": float("inf"),
        "relative_error_p50": float("inf"),
        "relative_error_p90": float("inf"),
        "relative_error_p95": float("inf"),
        "relative_error_p99": float("inf"),
        "relative_error_p995": float("inf"),
        "fraction_relative_error_le_0_1": 0.0,
        "fraction_relative_error_le_tau": 0.0,
        "near_zero_target_fraction": 0.0,
        "num_valid_points": 0,
        "eval_error": str(reason),
    }


def compute_output_base_metrics(y_pred, y, acc_tau=0.1, keep_frac=0.95):
    """LLM-SRBench/RESTART-style output metrics.

    - NMSE is MSE / Var(y), capped at 100 for robust aggregation.
    - Strict Acc@0.1 succeeds iff the all-point strict maximum relative error
      is <= 0.1. This matches the strict RESTART/LLM-SRBench interpretation.
    - The drop-worst-5% variant is retained only as best95 diagnostic.
    """
    try:
        y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
        y = np.asarray(y, dtype=float).reshape(-1)
    except Exception as exc:
        return _failure_metrics(f"array_cast_failed: {exc}")

    valid_idx = np.isfinite(y_pred) & np.isfinite(y)
    y_pred = y_pred[valid_idx]
    y = y[valid_idx]

    if len(y_pred) == 0 or len(y) == 0:
        return _failure_metrics("no_finite_predictions")

    if y_pred.shape[0] != y.shape[0]:
        n = min(y_pred.shape[0], y.shape[0])
        if n <= 0:
            return _failure_metrics(f"shape_mismatch: pred={y_pred.shape} y={y.shape}")
        y_pred = y_pred[:n]
        y = y[:n]

    mse = float(np.mean((y - y_pred) ** 2))
    var = float(np.mean((y - np.mean(y)) ** 2))
    denom_var = max(var, 1e-12)
    raw_nmse = float(mse / denom_var)
    nmse = float(min(raw_nmse, 100.0))

    centered_ss = float(np.sum((y - y.mean()) ** 2))
    r2 = float(1.0 - (np.sum((y - y_pred) ** 2) / max(centered_ss, 1e-12)))

    try:
        kdt = float(kendalltau(y, y_pred)[0])
    except Exception:
        kdt = float("nan")
    try:
        mape = float(mean_absolute_percentage_error(y, y_pred))
    except Exception:
        mape = float("inf")

    denom = np.maximum(np.abs(y), 1e-12)
    relative_error = np.abs(y_pred - y) / denom
    raw_max_relative_error = float(np.max(relative_error)) if len(relative_error) else float("inf")
    strict_max_relative_error = raw_max_relative_error
    relative_error_p50 = float(np.percentile(relative_error, 50)) if len(relative_error) else float("inf")
    relative_error_p90 = float(np.percentile(relative_error, 90)) if len(relative_error) else float("inf")
    relative_error_p95 = float(np.percentile(relative_error, 95)) if len(relative_error) else float("inf")
    relative_error_p99 = float(np.percentile(relative_error, 99)) if len(relative_error) else float("inf")
    relative_error_p995 = float(np.percentile(relative_error, 99.5)) if len(relative_error) else float("inf")
    fraction_relative_error_le_tau = float(np.mean(relative_error <= float(acc_tau))) if len(relative_error) else 0.0
    fraction_relative_error_le_0_1 = float(np.mean(relative_error <= 0.1)) if len(relative_error) else 0.0
    near_zero_target_fraction = float(np.mean(np.abs(y) <= 1e-8)) if len(y) else 0.0


    if len(relative_error) > 1:
        k = max(1, int(np.ceil(float(keep_frac) * len(relative_error))))
        keep_idx = np.argsort(relative_error)[:k]
        rel_kept = relative_error[keep_idx]
        y_kept = y[keep_idx]
        pred_kept = y_pred[keep_idx]
    else:
        rel_kept = relative_error
        y_kept = y
        pred_kept = y_pred

    max_relative_error = float(np.max(rel_kept)) if len(rel_kept) else float("inf")
    # Strict protocol: all valid points must satisfy the relative-error threshold.
    acc_0_1 = float(strict_max_relative_error <= float(acc_tau))

    best95_mse = float(np.mean((y_kept - pred_kept) ** 2)) if len(y_kept) else float("inf")
    best95_var = float(np.mean((y_kept - np.mean(y_kept)) ** 2)) if len(y_kept) else 0.0
    restart_best95_nmse = float(min(best95_mse / max(best95_var, 1e-12), 100.0))
    restart_best95_acc_0_1 = float(max_relative_error <= float(acc_tau))

    return {
        "mse": mse,
        "nmse": nmse,
        "raw_nmse": raw_nmse,
        "r2": r2,
        "kdt": kdt,
        "mape": mape,
        "acc_0_1": acc_0_1,
        "restart_nmse": nmse,
        "restart_acc_0_1": acc_0_1,
        "restart_best95_nmse": restart_best95_nmse,
        "restart_best95_acc_0_1": restart_best95_acc_0_1,
        "max_relative_error": max_relative_error,
        "raw_max_relative_error": raw_max_relative_error,
        "strict_max_relative_error": strict_max_relative_error,
        "relative_error_p50": relative_error_p50,
        "relative_error_p90": relative_error_p90,
        "relative_error_p95": relative_error_p95,
        "relative_error_p99": relative_error_p99,
        "relative_error_p995": relative_error_p995,
        "fraction_relative_error_le_0_1": fraction_relative_error_le_0_1,
        "fraction_relative_error_le_tau": fraction_relative_error_le_tau,
        "near_zero_target_fraction": near_zero_target_fraction,
        "num_valid_points": int(len(y_pred)),
        "num_points_after_95pct_filter": int(len(rel_kept)),
    }


class EvaluationPipeline:
    def __init__(self):
        pass

    def run_and_evaluate(self, searcher: BaseSearcher, problem: Problem):
        start_time = time.time()
        search_results: List[SearchResult] = searcher.discover(problem.create_task()) or []
        search_time = time.time() - start_time

        X_id = problem.test_samples[:, 1:]
        y_id = problem.test_samples[:, 0]

        X_ood = y_ood = None
        if problem.ood_test_samples is not None:
            X_ood = problem.ood_test_samples[:, 1:]
            y_ood = problem.ood_test_samples[:, 0]

        outs = []
        for result in search_results:
            equation = result.equation
            lambda_fn = equation.lambda_format

            try:
                id_pred = lambda_fn(X_id)
                id_output_base_metrics = compute_output_base_metrics(id_pred, y_id)
            except Exception as exc:
                id_output_base_metrics = _failure_metrics(f"id_eval_failed: {exc}")

            ood_output_base_metrics = None
            if X_ood is not None:
                try:
                    ood_pred = lambda_fn(X_ood)
                    ood_output_base_metrics = compute_output_base_metrics(ood_pred, y_ood)
                except Exception as exc:
                    ood_output_base_metrics = _failure_metrics(f"ood_eval_failed: {exc}")

            outs.append({
                "search_result": result,
                "search_time": search_time,
                "id_metrics": id_output_base_metrics,
                "ood_metrics": ood_output_base_metrics,
            })

        return outs

    def evaluate_problems(self,
                          problems: List[Problem],
                          searcher: BaseSearcher,
                          output_dir,
                          result_file_subfix=""):

        output_dir = Path(output_dir)
        output_file_path = output_dir / f"results{result_file_subfix}.jsonl"
        if output_dir.exists():
            visited_eqids = self.load_visited_problems(output_dir)
        else:
            visited_eqids = []

        for problem in problems:
            if problem.equation_idx in visited_eqids:
                print("Skipping problem: ", problem.equation_idx, f"(gt: {problem.gt_equation.expression})")
                continue

            print("Finding equation for problem: ", problem.equation_idx, f"(gt: {problem.gt_equation.expression})")
            outs = self.run_and_evaluate(searcher, problem)

            log_data = {
                'equation_id': problem.equation_idx,
                'gt_equation': problem.gt_equation.expression,
                'num_datapoints': len(problem.train_samples),
                'num_eval_datapoints': len(problem.test_samples),
            }
            eval_results = []
            top_level_planner_fields = {
                'planner_valid_candidates': None,
                'planner_input_candidates': None,
                'planner_max_disagreement': None,
                'planner_mean_disagreement': None,
                'planner_top_score': None,
                'planner_selected_points': None,
                'planner_selected_rationale': None,
                'planner_grid_coverage_summary': None,
            }
            for idx, out in enumerate(outs):
                eq_str_format = out['search_result'].equation.expression
                eq_program_format = out['search_result'].equation.program_format
                aux = dict(out['search_result'].aux or {}) if isinstance(out['search_result'].aux, dict) else {}

                planner_sources = [
                    aux,
                    aux.get('planner_diagnostics', {}) if isinstance(aux.get('planner_diagnostics', {}), dict) else {},
                    aux.get('pcpi_feedback_pack', {}).get('planner_diagnostics', {}) if isinstance(aux.get('pcpi_feedback_pack', {}), dict) else {},
                    aux.get('closed_loop_first_step_metrics', {}),
                    aux.get('pcpi_metrics', {}),
                ]
                for key in top_level_planner_fields:
                    if top_level_planner_fields[key] is not None:
                        continue
                    for src in planner_sources:
                        if isinstance(src, dict) and key in src and src.get(key) is not None:
                            top_level_planner_fields[key] = src.get(key)
                            break

                id_metrics = dict(out['id_metrics'] or {})
                ood_metrics = dict(out['ood_metrics'] or {}) if out['ood_metrics'] is not None else None

                eval_results.append({
                    'search_time': out['search_time'],
                    'discovered_equation': eq_str_format,
                    'discovered_program': eq_program_format,
                    'id_metrics': id_metrics,
                    'ood_metrics': ood_metrics,
                    'id_nmse': id_metrics.get('restart_nmse', id_metrics.get('nmse')),
                    'id_acc@0.1': id_metrics.get('restart_acc_0_1', id_metrics.get('acc_0_1')),
                    'ood_nmse': None if ood_metrics is None else ood_metrics.get('restart_nmse', ood_metrics.get('nmse')),
                    'ood_acc@0.1': None if ood_metrics is None else ood_metrics.get('restart_acc_0_1', ood_metrics.get('acc_0_1')),
                    'id_strict_max_relative_error': id_metrics.get('strict_max_relative_error', id_metrics.get('raw_max_relative_error', id_metrics.get('max_relative_error'))),
                    'id_relative_error_p95': id_metrics.get('relative_error_p95'),
                    'id_relative_error_p99': id_metrics.get('relative_error_p99'),
                    'id_fraction_relative_error_le_0.1': id_metrics.get('fraction_relative_error_le_0_1'),
                    'ood_strict_max_relative_error': None if ood_metrics is None else ood_metrics.get('strict_max_relative_error', ood_metrics.get('raw_max_relative_error', ood_metrics.get('max_relative_error'))),
                    'ood_relative_error_p95': None if ood_metrics is None else ood_metrics.get('relative_error_p95'),
                    'ood_relative_error_p99': None if ood_metrics is None else ood_metrics.get('relative_error_p99'),
                    'ood_fraction_relative_error_le_0.1': None if ood_metrics is None else ood_metrics.get('fraction_relative_error_le_0_1'),
                    **aux
                })
            log_data.update({k: v for k, v in top_level_planner_fields.items() if v is not None})
            log_data['eval_results'] = eval_results
            with open(output_file_path, mode='a', encoding='utf-8') as f:
                f.write(json.dumps(log_data, allow_nan=True) + "\n")

            visited_eqids.append(problem.equation_idx)

    @property
    def name(self):
        raise NotImplementedError

    def load_visited_problems(self, output_dir):
        result_files = list(Path(output_dir).glob("results*.jsonl"))
        visited = []
        if len(result_files) > 0:
            for result_file in result_files:
                with open(result_file, 'r', encoding='utf-8') as f:
                    for line in f.readlines():
                        if line.strip():
                            visited.append(json.loads(line)['equation_id'])
            visited = list(set(visited))
        return visited
