from dataclasses import dataclass
from pathlib import Path
from typing import Optional


@dataclass(frozen=True)
class ChallengePaths:
    """Canonical paths into the official student_resource directory."""

    student_resource: Path
    train_dir: Path
    test_dir: Path
    output_dir: Path

    @classmethod
    def from_student_resource(
        cls,
        student_resource: str | Path,
        output_dir: str | Path = "outputs/final",
    ) -> "ChallengePaths":
        root = Path(student_resource).expanduser().resolve()
        return cls(
            student_resource=root,
            train_dir=root / "dataset" / "train",
            test_dir=root / "dataset" / "test",
            output_dir=Path(output_dir).expanduser().resolve(),
        )

    def train_file(self, name: str) -> Path:
        return self.train_dir / name

    def test_file(self, name: str) -> Path:
        return self.test_dir / name


REQUIRED_COLUMNS = ["entity_id", "business_name", "business_address", "country"]
GROUND_TRUTH_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
MATCHING_COLUMNS = ["source1_entity_id", "matched_entity_ids"]
CANDIDATE_COLUMNS = ["source1_entity_id", "candidate_entity_ids"]
