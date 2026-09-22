#!/usr/bin/env python3
"""Read-only, standard-library verification of the released compact evidence.

Run ``python verify.py`` from any directory. This verifies the root manifest and
recomputes selected results from the committed tables; it does not rerun training.
Validation uses explicit exceptions and remains active under Python -O and -OO.
"""

import argparse
import csv
import hashlib
import itertools
import math
from pathlib import Path, PurePosixPath
import re
import statistics
import sys
from collections import Counter, defaultdict


class VerificationError(ValueError):
    """A released artifact violates a checked invariant."""


def require(condition, message):
    if not condition:
        raise VerificationError(message)


def number(value):
    result = float(value)
    require(math.isfinite(result), f"Non-finite number: {value!r}")
    return result


def close(actual, expected, label):
    actual, expected = number(actual), number(expected)
    require(math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-12),
            f"{label}: recorded {actual!r}, recomputed {expected!r}")


def verify_manifest(root):
    """Reject unsafe/duplicate paths and verify every listed file's bytes.

    The checked manifest is an integrity inventory, not a signature establishing
    the origin of the data. Files not listed in it are not covered by this check.
    """
    root = Path(root).resolve()
    seen = set()
    manifest = root / "SHA256SUMS.txt"
    require(not manifest.is_symlink(), "The root manifest must not be a symlink")
    for line_number, line in enumerate(manifest.read_text().splitlines(), 1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-fA-F]{64}) [ *](.+)", line)
        require(match is not None, f"Invalid manifest line {line_number}")
        digest, name = match.groups()
        path = PurePosixPath(name)
        require(not path.is_absolute() and ".." not in path.parts
                and "\\" not in name and ":" not in name
                and not any(ord(character) < 32 for character in name)
                and bool(path.parts), f"Unsafe manifest path: {name!r}")
        normalized = path.as_posix()
        require(normalized != "SHA256SUMS.txt", "Manifest cannot hash itself")
        require(path.suffix.lower() not in {".tex", ".bib", ".sty", ".cls"},
                f"Disallowed LaTeX source in Markdown repository: {normalized}")
        require(re.fullmatch(r"(?:paper|manuscript)(?:[_.-].*)?\.pdf", path.name.lower()) is None,
                f"Disallowed PDF manuscript in Markdown repository: {normalized}")
        require(normalized not in seen, f"Duplicate manifest path: {normalized}")
        seen.add(normalized)
        target = root
        for part in path.parts:
            target = target / part
            require(not target.is_symlink(), f"Symlink in manifest path: {name}")
        require(target.is_file(), f"Missing manifest file: {name}")
        actual = hashlib.sha256()
        with target.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                actual.update(chunk)
        require(actual.hexdigest() == digest.lower(), f"SHA256 mismatch: {name}")
    require(bool(seen), "The root manifest is empty")
    return seen


def read_csv(path):
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        header = reader.fieldnames
        require(bool(header) and len(header) == len(set(header)),
                f"Missing or duplicate CSV headers: {path}")
        rows = list(reader)
    require(bool(rows), f"Empty CSV: {path}")
    require(all(None not in row and None not in row.values() for row in rows),
            f"Malformed CSV row: {path}")
    return rows


def index_rows(rows, fields, label):
    result = {}
    for row in rows:
        key = tuple(row[field] for field in fields)
        require(key not in result, f"Duplicate {label}: {key}")
        result[key] = row
    return result


def exact_sign_flip(values, strata=None):
    """Enumerate the two-sided test, with equal task weights if strata are given."""
    values = [number(value) for value in values]
    require(0 < len(values) <= 20, "Exact test needs 1 to 20 finite values")
    if strata is None:
        values = [value for value in values if value != 0]
        if not values:
            return 1.0
        weights = [1 / len(values)] * len(values)
    else:
        require(len(strata) == len(values), "Strata/value length mismatch")
        counts = Counter(strata)
        weights = [1 / (len(counts) * counts[label]) for label in strata]
    weighted = [weight * value for weight, value in zip(weights, values)]
    observed = abs(math.fsum(weighted))
    exceed = sum(
        abs(math.fsum(sign * value for sign, value in zip(signs, weighted)))
        >= observed - 1e-15
        for signs in itertools.product((-1, 1), repeat=len(values))
    )
    return exceed / (2 ** len(values))


def average_ranks(values):
    """One-based average ranks, including exact ties."""
    counts = Counter(values)
    rank_map, offset = {}, 0
    for value, count in sorted(counts.items()):
        rank_map[value] = offset + (count + 1) / 2
        offset += count
    return [rank_map[value] for value in values]


def centered_moments(x, y):
    mx, my = statistics.mean(x), statistics.mean(y)
    xx = math.fsum((value - mx) ** 2 for value in x)
    yy = math.fsum((value - my) ** 2 for value in y)
    xy = math.fsum((a - mx) * (b - my) for a, b in zip(x, y))
    require(xx > 0 and yy > 0, "Constant Stage4 predictor or target")
    return mx, my, xx, yy, xy


def verify_stage4(root):
    folder = Path(root) / "evidence/stage4/aggregate"
    rows = read_csv(folder / "stage4_all_layer_summaries.csv")
    indexed = index_rows(rows, ("condition", "seed", "layer"), "Stage4 layer")
    conditions = {"hard_knee", "sample_limited"}
    seeds = {"101", "103", "107", "109", "113"}
    require(set(indexed) == {(condition, seed, str(layer)) for condition in conditions
                            for seed in seeds for layer in range(48)}, "Stage4 must contain 10 runs x 48 layers")
    target, predictor = "near_best_rank_gap_0.1", "gradient_effective_rank"
    primary = [row for row in read_csv(folder / "stage4_key_table.csv")
               if row["target"] == target and row["predictor"] == predictor]
    summary = index_rows(primary, ("condition",), "Stage4 primary summary")
    require(set(summary) == {(condition,) for condition in conditions}, "Stage4 primary summary coverage mismatch")
    output = []
    for condition in sorted(conditions):
        scores = []
        for seed in sorted(seeds):
            group = [indexed[(condition, seed, str(layer))] for layer in range(48)]
            require(all(row["target_valid"] == "True" for row in group), "Invalid Stage4 target")
            raw_x, raw_y = [number(row[predictor]) for row in group], [number(row[target]) for row in group]
            require(all(value >= 0 for value in raw_x) and all(value > 0 for value in raw_y), "Invalid Stage4 log input")
            x, y = [math.log2(1 + value) for value in raw_x], [math.log2(value) for value in raw_y]
            mx, my, xx, yy, xy = centered_moments(x, y)
            residuals = [b - my - xy / xx * (a - mx) for a, b in zip(x, y)]
            residual_ss = math.fsum(value ** 2 for value in residuals)
            _, _, rank_xx, rank_yy, rank_xy = centered_moments(average_ranks(raw_x), average_ranks(raw_y))
            # PRESS gives the same residuals as refitting this intercept+slope model
            # after omitting each layer, without numerical solver dependencies.
            press = math.fsum((residual / (1 - 1 / len(x) - (value - mx) ** 2 / xx)) ** 2
                              for value, residual in zip(x, residuals))
            scores.append((1 - residual_ss / yy, rank_xy / math.sqrt(rank_xx * rank_yy),
                           math.sqrt(residual_ss / len(x)), 1 - press / yy))
        means = [statistics.mean(score[column] for score in scores) for column in range(4)]
        row = summary[(condition,)]
        require(row["target_transform"] == "log2(rank)" and row["predictor_transform"] == "log2(1+s)",
                "Wrong Stage4 primary transforms")
        for field, value in (("n_runs", 5), ("mean_r2", means[0]), ("mean_spearman", means[1]),
                             ("mean_rmse_log2", means[2]), ("sem_r2", statistics.stdev(s[0] for s in scores) / math.sqrt(5)),
                             ("sem_spearman", statistics.stdev(s[1] for s in scores) / math.sqrt(5))):
            close(row[field], value, "Stage4 " + condition + " " + field)
        rounded_loo = {"hard_knee": 0.7547, "sample_limited": 0.6617}[condition]
        require(abs(means[3] - rounded_loo) <= 0.00005, "Stage4 leave-one-out R2 differs from reported rounded value")
        output.append(f"{condition} R2={means[0]:.10f}, rho={means[1]:.10f}, LOO R2={means[3]:.10f}")
    return "Stage4: 480 layers; " + "; ".join(output)


REAL_SEEDS = {
    "cattn_cfc": {"101", "103", "107", "109", "113", "127", "131", "137"},
    "attnproj": {"101", "103", "107"},
}
REAL_STRATEGIES = {
    "uniform_r4", "uniform_identity_control", "gradient_norm",
    "spectral_effective", "eva_activation", "fim_gradient_variance",
    "gora_sensitivity",
}


def verify_real_lora(root):
    folder = Path(root) / "evidence/real_lora/aggregate"
    rows = read_csv(folder / "all_results.csv")
    results = index_rows(rows, ("suite_id", "seed", "strategy"), "real-LoRA result")
    expected = {(suite, seed, strategy) for suite, seeds in REAL_SEEDS.items()
                for seed in seeds for strategy in REAL_STRATEGIES}
    require(set(results) == expected, "Real-LoRA must contain 11 runs x 7 strategies")
    run_names = {}
    for (suite, seed, strategy), row in results.items():
        require(row["suite_role"] == ("primary" if suite == "cattn_cfc" else "boundary"),
                "Real-LoRA suite role mismatch")
        identity = (suite, seed)
        require(identity not in run_names or run_names[identity] == row["run_id"],
                "Real-LoRA strategies disagree on run identity")
        run_names[identity] = row["run_id"]
        for field in ("initial_val_loss", "final_val_loss", "val_loss_delta", "perplexity"):
            number(row[field])
        close(row["val_loss_delta"], number(row["final_val_loss"]) - number(row["initial_val_loss"]),
              "real-LoRA loss change")
        close(row["trainable_params"], 331776 if suite == "cattn_cfc" else 405504,
              "real-LoRA exact parameter cost")
        close(row["budget_residual"], 0, "real-LoRA budget residual")
        close(row["n_train_steps"], 200, "real-LoRA training steps")
        require(row["adapter_activity_passed"] == "True", "Recorded adapter activity failed")
        for field in ("first_step_adapter_gradient_l2", "adapter_parameter_delta_l2",
                      "effective_update_frobenius_l2"):
            require(number(row[field]) > 0, f"Nonpositive recorded adapter activity: {field}")
    require(len(set(run_names.values())) == 11, "Real-LoRA run IDs must be unique")
    for suite, seed in run_names:
        reference = results[(suite, seed, "uniform_r4")]
        control = results[(suite, seed, "uniform_identity_control")]
        require(control["assignment_sha256"] == reference["assignment_sha256"], "Identity-control assignment mismatch")
        for strategy in REAL_STRATEGIES:
            row = results[(suite, seed, strategy)]
            close(row["initial_val_loss"], reference["initial_val_loss"], "shared real-LoRA initial loss")
            if row["assignment_sha256"] == reference["assignment_sha256"]:
                for field in ("final_val_loss", "val_loss_delta"):
                    close(row[field], reference[field], "identical-assignment " + field)
                for field in ("train_trace_sha256", "initial_adapter_state_sha256", "final_adapter_state_sha256"):
                    require(row[field] == reference[field], f"Identical-assignment {field} mismatch")

    paired = index_rows(read_csv(folder / "paired_deltas.csv"),
                        ("suite_id", "seed", "candidate_strategy"), "real-LoRA pair")
    candidates = REAL_STRATEGIES - {"uniform_r4", "uniform_identity_control"}
    require(set(paired) == {key for key in expected if key[2] in candidates},
            "Real-LoRA paired table must contain all 55 candidate comparisons")
    for (suite, seed, strategy), pair in paired.items():
        candidate, reference = results[(suite, seed, strategy)], results[(suite, seed, "uniform_r4")]
        require(pair["run_id"] == candidate["run_id"]
                and pair["reference_strategy"] == "uniform_r4", "Real-LoRA pair identity mismatch")
        for metric in ("final_val_loss", "val_loss_delta", "perplexity"):
            close(pair["candidate_minus_reference_" + metric],
                  number(candidate[metric]) - number(reference[metric]), f"real-LoRA paired {metric}")
        for prefix, source in (("candidate", candidate), ("reference", reference)):
            close(pair[prefix + "_trainable_params"], source["trainable_params"], "paired parameter cost")
            require(pair[prefix + "_assignment_sha256"] == source["assignment_sha256"],
                    "Paired assignment digest mismatch")
        close(pair["parameter_difference"], 0, "real-LoRA paired cost gap")
        identical = candidate["assignment_sha256"] == reference["assignment_sha256"]
        require(pair["assignments_identical"] == str(identical), "Assignment equality mismatch")

    primary = index_rows(read_csv(folder / "primary_seed_deltas.csv"),
                         ("suite_id", "seed", "candidate_strategy"), "real-LoRA primary run")
    require(set(primary) == {("cattn_cfc", seed, "spectral_effective")
                             for seed in REAL_SEEDS["cattn_cfc"]}, "Real-LoRA primary run set mismatch")
    for key, row in primary.items():
        require(row == paired[key], f"Primary and paired real-LoRA rows disagree: {key}")
    values = [number(row["candidate_minus_reference_final_val_loss"]) for row in primary.values()]
    summaries = read_csv(folder / "primary_analysis.csv")
    require(len(summaries) == 1, "Expected one real-LoRA primary summary")
    summary = summaries[0]
    require(summary["suite_id"] == "cattn_cfc" and summary["candidate_strategy"] == "spectral_effective"
            and summary["reference_strategy"] == "uniform_r4", "Wrong real-LoRA primary contrast")
    mean, p_value = statistics.mean(values), exact_sign_flip(values)
    checks = {"n_independent_runs": 8, "n_nonzero_deltas": sum(value != 0 for value in values),
              "mean_loss_delta": mean, "median_loss_delta": statistics.median(values),
              "sd_loss_delta": statistics.stdev(values),
              "sem_loss_delta": statistics.stdev(values) / math.sqrt(len(values)),
              "exact_sign_flip_p_two_sided": p_value,
              "wins": sum(value < 0 for value in values), "ties": values.count(0),
              "losses": sum(value > 0 for value in values)}
    for field, expected_value in checks.items():
        close(summary[field], expected_value, "real-LoRA primary " + field)
    return f"Real LoRA: 77 result rows, 55 paired comparisons; primary n=8, mean={mean:.12g}, p={p_value:.12g}"


TRANSFORMER_SEEDS = {"101", "103", "107", "109", "113"}
TRANSFORMER_BUDGETS = {
    "associative_recall": {"512", "1024", "1536", "2048", "3072", "4096", "6144"},
    "modular": {"512", "1024", "1536", "2048", "3072", "4096"},
}


def verify_transformer(root):
    folder = Path(root) / "evidence/transformer/aggregate"
    rows = read_csv(folder / "run_level_budget_results.csv")
    fields = ("task_family", "run", "scaling_mode", "condition_id", "budget")
    results = index_rows(rows, fields, "transformer budget result")
    require(len(results) == 1620, "Expected 1620 transformer run/budget/condition rows")
    references, candidates = {}, {}
    for key, row in results.items():
        for field in ("actual_cost", "requested_budget", "final_val_loss", "final_val_accuracy"):
            number(row[field])
        close(row["requested_budget"], row["budget"], "transformer requested budget")
        require(0 <= number(row["actual_cost"]) <= number(row["requested_budget"]),
                "Transformer actual cost outside its nonnegative budget cap")
        close(row["n_adaptation_replicates"], 3, "transformer adaptation replicate count")
        close(row["diverged_count"], 0, "transformer recorded divergence count")
        require(0 <= number(row["final_val_accuracy"]) <= 1, "Transformer accuracy outside [0, 1]")
        if row["comparison_role"] == "exact_cost_baseline":
            ref_key = key[:3] + (number(row["actual_cost"]),)
            require(ref_key not in references, f"Duplicate exact-cost reference: {ref_key}")
            references[ref_key] = row
        elif row["comparison_role"] == "candidate":
            candidates[key] = row
    rules = {"effective_rank", "gradient_norm", "marginal_gain_edge", "marginal_gain_soft", "soft_dimension"}
    scales = {"fixed_update_scale", "rslora", "standard"}
    candidate_identities = index_rows(candidates.values(),
        ("task_family", "seed", "scaling_mode", "condition_id", "budget"), "transformer candidate")
    require(set(candidate_identities) == {
        (task, seed, scale, rule, budget) for task, budgets in TRANSFORMER_BUDGETS.items()
        for seed in TRANSFORMER_SEEDS for scale in scales for rule in rules for budget in budgets
    }, "Transformer candidate task/seed/scaling/rule/budget grid mismatch")
    recorded_deltas = index_rows(read_csv(folder / "run_level_deltas_exact_cost.csv"),
                                 fields, "transformer paired delta")
    require(set(recorded_deltas) == set(candidates), "Transformer paired delta coverage mismatch")
    primary_budgets = defaultdict(list)
    for key, candidate in candidates.items():
        cost = number(candidate["actual_cost"])
        ref_key = key[:3] + (cost,)
        require(ref_key in references, f"Missing exact-cost reference: {ref_key}")
        reference, recorded = references[ref_key], recorded_deltas[key]
        require(reference["seed"] == candidate["seed"] == recorded["seed"],
                "Transformer pair seed mismatch")
        loss = number(candidate["final_val_loss"]) - number(reference["final_val_loss"])
        accuracy = number(candidate["final_val_accuracy"]) - number(reference["final_val_accuracy"])
        for field, value in (("loss_delta_vs_exact_uniform", loss), ("acc_delta_vs_exact_uniform", accuracy),
                             ("actual_cost", cost), ("reference_cost", cost), ("cost_gap", 0),
                             ("n_adaptation_replicates", 3)):
            close(recorded[field], value, "transformer paired " + field)
        if candidate["scaling_mode"] == "fixed_update_scale" and candidate["condition_id"] == "soft_dimension":
            primary_budgets[(candidate["task_family"], candidate["run"], candidate["seed"])].append(
                (candidate["budget"], loss, accuracy, cost))
    require(len(primary_budgets) == 10, "Expected 10 distinct transformer primary runs")
    primary = index_rows(read_csv(folder / "primary_run_deltas.csv"),
                         ("task_family", "run", "seed"), "transformer primary run")
    require(set(primary) == set(primary_budgets), "Transformer primary run coverage mismatch")
    task_values, task_accuracies = defaultdict(list), defaultdict(list)
    for key, budget_rows in primary_budgets.items():
        task = key[0]
        require({row[0] for row in budget_rows} == TRANSFORMER_BUDGETS[task],
                "Transformer primary budget set mismatch")
        row = primary[key]
        require(row["scaling_mode"] == "fixed_update_scale" and row["condition_id"] == "soft_dimension",
                "Wrong transformer primary contrast")
        loss, accuracy, cost = [statistics.mean(entry[column] for entry in budget_rows) for column in (1, 2, 3)]
        for field, value in (("loss_delta_vs_exact_uniform", loss), ("acc_delta_vs_exact_uniform", accuracy),
                             ("mean_actual_cost", cost), ("n_budgets", len(budget_rows)),
                             ("mean_adaptation_replicates", 3)):
            close(row[field], value, "transformer primary run " + field)
        task_values[task].append(loss)
        task_accuracies[task].append(accuracy)
    by_task = index_rows(read_csv(folder / "primary_analysis_by_task.csv"), ("task_family",), "task summary")
    require(set(by_task) == {(task,) for task in TRANSFORMER_BUDGETS}, "Transformer task summary coverage mismatch")
    for task, values in task_values.items():
        for field, value in (("mean_loss_delta", statistics.mean(values)), ("n_independent_runs", 5),
                             ("exact_sign_flip_p_two_sided", exact_sign_flip(values)),
                             ("run_wins_loss", sum(value < 0 for value in values)), ("run_ties_loss", values.count(0))):
            close(by_task[(task,)][field], value, "transformer task " + task + " " + field)
    values = [value for group in task_values.values() for value in group]
    strata = [task for task, group in task_values.items() for _ in group]
    mean = statistics.mean(statistics.mean(group) for group in task_values.values())
    p_value = exact_sign_flip(values, strata)
    summaries = read_csv(folder / "primary_analysis.csv")
    require(len(summaries) == 1, "Expected one transformer primary summary")
    summary = summaries[0]
    require(summary["reference_rule"] == "uniform_exact_cost"
            and summary["scaling_mode"] == "fixed_update_scale" and summary["condition_id"] == "soft_dimension"
            and summary["task_weighting"] == "equal_weight_across_task_families", "Wrong transformer primary estimand")
    checks = {"mean_loss_delta": mean, "exact_sign_flip_p_two_sided": p_value,
              "n_task_families": 2, "n_independent_runs": 10, "min_runs_per_task": 5, "max_runs_per_task": 5,
              "min_budgets_per_run": 6, "max_budgets_per_run": 7,
              "exact_sign_flip_n_units": 10, "exact_sign_flip_n_patterns": 1024,
              "run_wins_loss": sum(value < 0 for value in values), "run_ties_loss": values.count(0),
              "mean_accuracy_delta": statistics.mean(statistics.mean(group) for group in task_accuracies.values())}
    for field, value in checks.items():
        close(summary[field], value, "transformer primary " + field)
    return f"Transformer: 975 exact-cost comparisons; primary n=10, mean={mean:.12g}, p={p_value:.12g}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parent,
                        help="Repository root (default: directory containing verify.py)")
    args = parser.parse_args(argv)
    try:
        paths = verify_manifest(args.root)
        # Every compact table used by the calculations must also be hash-covered.
        required = {
            "stage4": ("stage4_all_layer_summaries", "stage4_key_table"),
            "real_lora": ("all_results", "paired_deltas", "primary_seed_deltas", "primary_analysis"),
            "transformer": ("run_level_budget_results", "run_level_deltas_exact_cost", "primary_run_deltas",
                            "primary_analysis_by_task", "primary_analysis"),
        }
        for family, names in required.items():
            for name in names:
                require(f"evidence/{family}/aggregate/{name}.csv" in paths,
                        f"Checked evidence missing from root manifest: {family}/{name}.csv")
        print(f"PASS: root SHA256 manifest ({len(paths)} files)")
        print("PASS: " + verify_stage4(args.root))
        print("PASS: " + verify_real_lora(args.root))
        print("PASS: " + verify_transformer(args.root))
    except (VerificationError, OSError, ValueError, KeyError, csv.Error) as error:
        print(f"FAIL: {error}", file=sys.stderr)
        return 1
    print("Scope: checks committed table integrity and selected arithmetic, including costs and exact p-values.")
    print("Limits: no training rerun, unavailable raw-state verification, bootstrap-interval reproduction, or novelty validation.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
