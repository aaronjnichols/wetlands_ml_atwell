from pathlib import Path
import sys

SRC = Path(__file__).resolve().parent / "src"
if SRC.exists() and str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from wetlands_ml_atwell.test_unet import *  # noqa: F401,F403


def main() -> None:
    from wetlands_ml_atwell.test_unet import main as _main

    _main()


if __name__ == "__main__":
    main()
