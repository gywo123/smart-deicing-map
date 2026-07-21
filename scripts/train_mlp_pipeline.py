"""실제 기상청 노면온도 MLP 학습과 지도 생성 실행 스크립트."""

from __future__ import annotations

import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
WORKSPACE_DIR = PROJECT_DIR.parent
sys.path.insert(0, str(WORKSPACE_DIR))
sys.path.insert(0, str(PROJECT_DIR))

from src.mlp_training_pipeline import train_mlp_pipeline  # noqa: E402
import scripts.main_pipeline as pipeline  # noqa: E402


def main() -> dict[str, object]:
    """3시간 후 노면온도 MLP 학습과 상대 제설 위험도 산출을 실행한다."""
    return train_mlp_pipeline(pipeline)


if __name__ == "__main__":
    main()
