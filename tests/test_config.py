import copy
import tomllib
import unittest
from pathlib import Path

from ot_micromr.config import load_runspec, validate_runspec
from ot_micromr.errors import ConfigError


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


def load_data(name: str) -> dict:
    path = REPOSITORY_ROOT / "cfg" / "experiments" / name
    return tomllib.loads(path.read_text(encoding="utf-8"))


class RunSpecTests(unittest.TestCase):
    def test_analytical_configs_validate(self) -> None:
        for name in ("ana_smoke_001.toml", "ana_fig3_001.toml"):
            with self.subTest(name=name):
                spec = load_runspec(REPOSITORY_ROOT / "cfg" / "experiments" / name)
                self.assertEqual(spec.values["schema_version"], "runspec-v1")
                self.assertEqual(len(spec.sha256), 64)

    def test_canonical_hash_is_deterministic(self) -> None:
        path = REPOSITORY_ROOT / "cfg" / "experiments" / "ana_smoke_001.toml"
        first = load_runspec(path)
        second = load_runspec(path)
        self.assertEqual(first.canonical_json, second.canonical_json)
        self.assertEqual(first.sha256, second.sha256)

    def test_runspec_is_deeply_immutable(self) -> None:
        spec = load_runspec(REPOSITORY_ROOT / "cfg" / "experiments" / "ana_smoke_001.toml")
        with self.assertRaises(TypeError):
            spec.values["experiment_id"] = "changed"
        with self.assertRaises(TypeError):
            spec.values["model"]["gamma_ratio"] = 0.5
        with self.assertRaises(TypeError):
            spec.values["claim_ids"][0] = "changed"

    def test_unknown_field_is_rejected(self) -> None:
        data = load_data("ana_smoke_001.toml")
        data["unexpected"] = True
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            validate_runspec(data)

    def test_unknown_nested_field_is_rejected(self) -> None:
        data = load_data("ana_smoke_001.toml")
        data["numerics"]["hidden_tolerance"] = 1e-5
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            validate_runspec(data)

    def test_missing_field_is_rejected(self) -> None:
        data = load_data("ana_smoke_001.toml")
        del data["model"]["gamma_ratio"]
        with self.assertRaisesRegex(ConfigError, "missing required fields"):
            validate_runspec(data)

    def test_nonfinite_float_is_rejected(self) -> None:
        data = load_data("ana_smoke_001.toml")
        data["model"]["gamma_ratio"] = float("inf")
        with self.assertRaisesRegex(ConfigError, "non-finite"):
            validate_runspec(data)

    def test_incompatible_mode_is_rejected(self) -> None:
        data = load_data("ana_smoke_001.toml")
        data["mode"] = "oracle-diagnostic"
        with self.assertRaisesRegex(ConfigError, "paper-faithful"):
            validate_runspec(data)

    def test_p3_simulation_contracts_validate(self) -> None:
        for name in (
            "sim_moments_002.toml",
            "sim_unbalanced_002.toml",
        ):
            with self.subTest(name=name):
                validate_runspec(load_data(name))

    def test_figure4_target_contract_validates(self) -> None:
        data = load_data("sim_fig4_002.toml")
        validate_runspec(data)
        data["evaluation"]["planned_strategy_seed_count"] = 29
        with self.assertRaisesRegex(ConfigError, "disagrees with strategy seeds"):
            validate_runspec(data)

    def test_empirical_target_contract_validates(self) -> None:
        spec = load_runspec(REPOSITORY_ROOT / "cfg" / "experiments" / "emp_data_001.toml")
        self.assertEqual(spec.values["track"], "empirical")
        self.assertEqual(
            spec.values["inputs"]["dataset_sha256"],
            "0e3a6d6e99586b72ccc237bde7f8df4c3651ba4bd4495b391d9a20771c0e3888",
        )

        data = load_data("emp_data_001.toml")
        data["strategy"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "strategy.enabled: EMP-DATA-001 requires false"):
            validate_runspec(data)

    def test_p6_synthetic_filter_contract_validates(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "filter_syn_001.toml"
        )
        self.assertEqual(spec.values["track"], "synthetic")
        self.assertEqual(spec.values["numerics"]["compute_device"], "cuda")

        data = load_data("filter_syn_001.toml")
        data["strategy"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "strategy.enabled: FILTER-SYN-001"):
            validate_runspec(data)

    def test_p6_empirical_filter_contract_and_dependencies_validate(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_filter_001.toml"
        )
        self.assertEqual(spec.values["evaluation"]["audit_date"], "2024-12-15")
        self.assertEqual(spec.values["inputs"]["validation_and_test_access"], "forbidden")

        data = load_data("emp_filter_001.toml")
        data["execution"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "execution.enabled: EMP-FILTER-001"):
            validate_runspec(data)

    def test_p6m_marked_filter_contracts_validate(self) -> None:
        synthetic = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "filter_mark_syn_001.toml"
        )
        empirical = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_filter_001.toml"
        )
        continuous = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_ct_001.toml"
        )
        self.assertEqual(synthetic.values["model"]["mark_contract"], "fixed_729_bucket_v1")
        self.assertEqual(empirical.values["evaluation"]["planned_blocks"], 288)
        self.assertEqual(continuous.values["numerics"]["hazard_primary_substeps"], 4)
        self.assertEqual(continuous.values["numerics"]["hazard_refinement_substeps"], 8)

        data = load_data("emp_mark_filter_001.toml")
        data["strategy"]["enabled"] = True
        with self.assertRaisesRegex(ConfigError, "strategy.enabled: EMP-MARK-FILTER-001"):
            validate_runspec(data)

    def test_p6d_factorized_filter_contract_validates(self) -> None:
        spec = load_runspec(
            REPOSITORY_ROOT / "cfg" / "experiments" / "emp_mark_fact_001.toml"
        )
        self.assertEqual(
            spec.values["model"]["factorization"],
            "renewal_clock_times_conditional_mark_v1",
        )
        self.assertEqual(spec.values["model"]["clock_history_events"], 200)
        data = load_data("emp_mark_fact_001.toml")
        data["model"]["clock_history_events"] = 201
        with self.assertRaisesRegex(ConfigError, "clock_history_events: expected 200"):
            validate_runspec(data)

    def test_unknown_simulation_field_is_rejected(self) -> None:
        data = load_data("sim_moments_002.toml")
        data["simulation"]["implicit_dt"] = 0.1
        with self.assertRaisesRegex(ConfigError, "unknown fields"):
            validate_runspec(data)

    def test_to_dict_returns_independent_mutable_copy(self) -> None:
        spec = load_runspec(REPOSITORY_ROOT / "cfg" / "experiments" / "ana_smoke_001.toml")
        copy_one = spec.to_dict()
        copy_two = copy.deepcopy(copy_one)
        copy_two["model"]["gamma_ratio"] = 9.0
        self.assertEqual(copy_one["model"]["gamma_ratio"], 0.4)


if __name__ == "__main__":
    unittest.main()
