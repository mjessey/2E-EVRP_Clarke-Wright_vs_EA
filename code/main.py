# main.py
# -------------------------------------------------------------------
#  Command-line launcher
#   • get file path from user (arg 1 or interactive prompt)
#   • parse instance
#   • brute-force search for a feasible optimum
#   • open GUI to visualise the instance + optimal routes
# -------------------------------------------------------------------

import sys
from pathlib import Path

from core.parser import Parser
from solvers.brute_force import BruteForce
from gui.gui import GUI


# -------------------------------------------------------------------
def ask_for_path() -> Path:
    """Repeatedly prompt until the user types an existing file path."""
    while True:
        p = input("Enter path to 2E-EVRP instance (or 'q' to quit): ").strip()
        if p.lower() in {"q", "quit", "exit"}:
            sys.exit(0)
        path = Path(p)
        if path.is_file():
            return path
        print(f"'{p}' is not a valid file – try again.\n")


# -------------------------------------------------------------------
def main() -> None:
    # 1) obtain path -------------------------------------------------
    if len(sys.argv) >= 2:
        instance_path = Path(sys.argv[1])
        if not instance_path.is_file():
            print(f"Error: '{instance_path}' is not a file.")
            instance_path = ask_for_path()
    else:
        instance_path = ask_for_path()

    # 2) parse -------------------------------------------------------
    try:
        data = Parser(instance_path).data
    except Exception as err:
        print("Parsing failed:", err)
        sys.exit(1)

    # 3) solve -------------------------------------------------------
    solver = BruteForce()
    result = solver.solve(data)

    best_sol = result["solution"]
    if best_sol is None:
        print("No feasible solution found.")
        return

    print("-------------------------------------------------")
    print(f"#EVs used        : {result['evs']}")
    print(f"Total distance   : {result['distance']:.2f}")

    # optional fields – print only if they exist
    if "routes" in result:
        print(f"Routes enumerated: {result['routes']}")
    if "milp_vars" in result:
        print(f"MILP variables   : {result['milp_vars']}")
    if "milp_time" in result and result["milp_time"] is not None:
        print(f"MILP solve time  : {result['milp_time']:.3f} s")
    print("-------------------------------------------------")

    # 4) visualise ---------------------------------------------------
    GUI(data, solution=best_sol).run()


# -------------------------------------------------------------------
if __name__ == "__main__":
    main()
