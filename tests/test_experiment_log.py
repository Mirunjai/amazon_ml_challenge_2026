import pandas as pd

from src.experiment_log import COLUMNS, ExperimentRecord, append_experiment, load_experiment_log


def test_load_missing_log_returns_empty_frame_with_correct_columns(tmp_path):
    path = tmp_path / "does_not_exist.csv"
    df = load_experiment_log(path)
    assert df.empty
    assert list(df.columns) == COLUMNS


def test_append_creates_file_with_header(tmp_path):
    path = tmp_path / "experiments.csv"
    record = ExperimentRecord(
        experiment_id="exp_001",
        features_version="baseline_v1",
        model="RandomForestClassifier",
        model_params={"n_estimators": 200, "random_state": 42},
        threshold=0.5,
        candidate_pairs=1000,
        positive_count=400,
        negative_count=600,
        candidate_recall=0.92,
        precision=0.81,
        recall=0.77,
        macro_f05=0.80,
        notes="first baseline run",
    )

    result_path = append_experiment(record, path)

    assert result_path == path
    df = pd.read_csv(path)
    assert list(df.columns) == COLUMNS
    assert len(df) == 1
    assert df.loc[0, "experiment_id"] == "exp_001"
    assert df.loc[0, "model"] == "RandomForestClassifier"
    assert df.loc[0, "candidate_pairs"] == 1000


def test_append_twice_accumulates_rows_without_duplicate_header(tmp_path):
    path = tmp_path / "experiments.csv"
    r1 = ExperimentRecord(experiment_id="exp_001", features_version="v1", model="RF")
    r2 = ExperimentRecord(experiment_id="exp_002", features_version="v1", model="ExtraTrees")

    append_experiment(r1, path)
    append_experiment(r2, path)

    df = load_experiment_log(path)
    assert len(df) == 2
    assert list(df["experiment_id"]) == ["exp_001", "exp_002"]


def test_optional_fields_default_sensibly(tmp_path):
    path = tmp_path / "experiments.csv"
    record = ExperimentRecord(experiment_id="exp_003", features_version="v1", model="RF")
    append_experiment(record, path)

    df = load_experiment_log(path)
    row = df.iloc[0]
    assert pd.isna(row["macro_f05"])
    assert pd.isna(row["candidate_recall"])
    assert row["notes"] == "" or pd.isna(row["notes"])


def test_model_params_round_trips_as_json(tmp_path):
    import json

    path = tmp_path / "experiments.csv"
    record = ExperimentRecord(
        experiment_id="exp_004",
        features_version="v1",
        model="RF",
        model_params={"n_estimators": 300, "max_depth": 8},
    )
    append_experiment(record, path)

    df = load_experiment_log(path)
    params = json.loads(df.loc[0, "model_params"])
    assert params == {"n_estimators": 300, "max_depth": 8}
