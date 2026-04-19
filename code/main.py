# -----------------------------------------------------------------
#  1. get a valid file path from CLI or from an interactive prompt
#  2. hand that path directly to Parser
#  3. launch the PyGame viewer
# -----------------------------------------------------------------

import sys
from pathlib import Path

from parser import Parser   # <- your Parser
from ui import UI        # <- PyGame viewer


def ask_for_path() -> Path:
    """Keep asking until the user supplies an existing file path."""
    while True:
        p = input("Enter path to 2E-EVRP instance file (or 'q' to quit): ").strip()
        if p.lower() in {"q", "quit", "exit"}:
            sys.exit(0)
        path = Path(p)
        if path.is_file():
            return path
        print(f"'{p}' does not exist or is not a file. Try again.\n")


def main() -> None:
    """
    Usage examples
    --------------
    python main.py                      # prompts for a path
    python main.py ../data/C101_C5x.txt # uses the argument directly
    """
    # -----------------------------------------------------------------
    # 1) obtain path
    # -----------------------------------------------------------------
    if len(sys.argv) >= 2:
        instance_path = Path(sys.argv[1])
        if not instance_path.is_file():
            print(f"Error: '{instance_path}' does not exist or is not a file.")
            instance_path = ask_for_path()
    else:
        instance_path = ask_for_path()

    # -----------------------------------------------------------------
    # 2) parse the file (pass PATH, not raw text!)
    # -----------------------------------------------------------------
    try:
        data = Parser(instance_path).data
    except Exception as err:
        print(f"Parsing failed: {err}")
        sys.exit(1)

    # -----------------------------------------------------------------
    # 3) launch the viewer
    # -----------------------------------------------------------------
    UI(data).run()


if __name__ == "__main__":
    main()
