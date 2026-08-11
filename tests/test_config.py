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
