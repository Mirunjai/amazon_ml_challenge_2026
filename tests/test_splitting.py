import pandas as pd
import pytest

from src.splitting import (
    assert_no_entity_leakage,
    entity_level_train_val_split,
    split_pairs_by_entity,
)


class TestEntityLevelSplit:
    def test_no_overlap(self):
        entity_ids = [f"S1-{i:05d}" for i in range(1000)]
        train_ids, val_ids = entity_level_train_val_split(entity_ids, val_fraction=0.2, seed=1)
        assert train_ids.isdisjoint(val_ids)

    def test_covers_every_entity_exactly_once(self):
        entity_ids = [f"S1-{i:05d}" for i in range(500)]
        train_ids, val_ids = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=7)
        assert train_ids | val_ids == set(entity_ids)
        assert len(train_ids) + len(val_ids) == len(set(entity_ids))

    def test_val_fraction_approximately_respected(self):
        entity_ids = [f"S1-{i:05d}" for i in range(10_000)]
        train_ids, val_ids = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=42)
        frac = len(val_ids) / len(entity_ids)
        assert frac == pytest.approx(0.15, abs=0.01)

    def test_reproducible_with_same_seed(self):
        entity_ids = [f"S1-{i:05d}" for i in range(300)]
        t1, v1 = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=42)
        t2, v2 = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=42)
        assert t1 == t2
        assert v1 == v2

    def test_different_seeds_generally_differ(self):
        entity_ids = [f"S1-{i:05d}" for i in range(300)]
        _, v1 = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=1)
        _, v2 = entity_level_train_val_split(entity_ids, val_fraction=0.15, seed=2)
        assert v1 != v2

    def test_reproducible_regardless_of_input_order(self):
        ids_a = [f"S1-{i:05d}" for i in range(200)]
        ids_b = list(reversed(ids_a))
        t1, v1 = entity_level_train_val_split(ids_a, val_fraction=0.2, seed=5)
        t2, v2 = entity_level_train_val_split(ids_b, val_fraction=0.2, seed=5)
        assert t1 == t2
        assert v1 == v2

    def test_deduplicates_repeated_entity_ids(self):
        """Passing an id multiple times (e.g. from a pairs table) should not
        double count it."""
        entity_ids = ["S1-1", "S1-1", "S1-2", "S1-2", "S1-3"]
        train_ids, val_ids = entity_level_train_val_split(entity_ids, val_fraction=0.34, seed=0)
        assert len(train_ids) + len(val_ids) == 3

    def test_invalid_val_fraction_raises(self):
        with pytest.raises(ValueError):
            entity_level_train_val_split(["S1-1", "S1-2"], val_fraction=0.0)
        with pytest.raises(ValueError):
            entity_level_train_val_split(["S1-1", "S1-2"], val_fraction=1.0)


class TestSplitPairsByEntity:
    def test_all_pairs_for_one_s1_stay_together(self):
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-A", "S1-A", "S1-A", "S1-B", "S1-B"],
                "candidate_entity_id": ["S2-1", "S2-2", "S3-1", "S2-3", "S3-2"],
            }
        )
        train_ids = {"S1-A"}
        val_ids = {"S1-B"}

        train_df, val_df = split_pairs_by_entity(pairs, train_ids, val_ids)

        assert set(train_df["source1_entity_id"]) == {"S1-A"}
        assert len(train_df) == 3
        assert set(val_df["source1_entity_id"]) == {"S1-B"}
        assert len(val_df) == 2

    def test_rows_outside_both_sets_are_dropped(self):
        pairs = pd.DataFrame(
            {
                "source1_entity_id": ["S1-A", "S1-C"],
                "candidate_entity_id": ["S2-1", "S2-2"],
            }
        )
        train_df, val_df = split_pairs_by_entity(pairs, {"S1-A"}, {"S1-B"})
        assert len(train_df) == 1
        assert len(val_df) == 0

    def test_end_to_end_with_real_split(self):
        """Feature/label alignment: splitting a labeled pairs frame by the
        entity-level split should leave every row's S1 id in the matching set,
        and no S1 id split across both."""
        entity_ids = [f"S1-{i:03d}" for i in range(50)]
        train_ids, val_ids = entity_level_train_val_split(entity_ids, val_fraction=0.2, seed=3)

        rows = []
        for eid in entity_ids:
            for c in range(3):
                rows.append({"source1_entity_id": eid, "candidate_entity_id": f"S2-{eid}-{c}", "label": c == 0})
        pairs = pd.DataFrame(rows)

        train_df, val_df = split_pairs_by_entity(pairs, train_ids, val_ids)

        assert set(train_df["source1_entity_id"]) <= train_ids
        assert set(val_df["source1_entity_id"]) <= val_ids
        assert set(train_df["source1_entity_id"]).isdisjoint(set(val_df["source1_entity_id"]))
        # every row preserved somewhere (feature/label alignment kept)
        assert len(train_df) + len(val_df) == len(pairs)


class TestAssertNoEntityLeakage:
    def test_passes_on_disjoint_sets(self):
        assert_no_entity_leakage({"S1-1", "S1-2"}, {"S1-3", "S1-4"})

    def test_raises_on_overlap(self):
        with pytest.raises(ValueError, match="leakage"):
            assert_no_entity_leakage({"S1-1", "S1-2"}, {"S1-2", "S1-3"})
