# -----------------------------------------------------------
#  A very small, dependency–free parser for text instances of
#  the Two–Echelon Electric Vehicle Routing Problem (2E-EVRP)
# -----------------------------------------------------------

from pathlib import Path
from typing import Dict, List, Any


class Parser:
    """
    Reads a 2E-EVRP data file (or string) and converts it into a
    dictionary that is convenient to use in algorithms.

        data = Parser(path_or_string).data
    """

    # ---------------------------------------------------------------------
    # public interface
    # ---------------------------------------------------------------------
    def __init__(self, source: str | Path) -> None:
        """
        source  –  either a path to a text file or the raw string itself
        The parsed result is stored in self.data
        """
        # 1.  get the text
        text = Path(source).read_text() if isinstance(source, Path) or Path(source).exists() else source

        # 2.  do the work
        self.data: Dict[str, Any] = {
            "nodes": {},          # id -> attributes
            "depots": [],         # list of ids
            "satellites": [],     # list of ids
            "stations": [],       # list of ids  (charging-/re-fuel stations)
            "customers": [],      # list of ids
            "params": {}          # global scalar parameters
        }
        self._parse(text)

    # ---------------------------------------------------------------------
    # internal helpers
    # ---------------------------------------------------------------------
    def _parse(self, text: str) -> None:
        # strip empty lines / comment-only lines
        lines = [ln.strip() for ln in text.splitlines() if ln.strip() and not ln.lstrip().startswith("#")]

        # -----------------------------------------------------------------
        # 1. node header  (the first non-empty line)
        # -----------------------------------------------------------------
        header: List[str] = lines[0].split()      # 'StringID   Type   x   …'
        numeric_cols = header[2:]                 # every field except StringID and Type is numeric

        # -----------------------------------------------------------------
        # 2. node records  (all lines whose first token length > 1)
        # -----------------------------------------------------------------
        idx = 1
        while idx < len(lines):
            tokens = lines[idx].split()
            first = tokens[0]

            # parameter section begins when the first token is ONE single
            # character (L, C, Q, r, g, v, …)
            if len(first) == 1:
                break

            entry = {header[i]: self._to_number(tokens[i]) for i in range(1, len(header))}
            ntype = entry["Type"]

            # store in the global dictionary ------------------------------------------------
            self.data["nodes"][first] = entry

            if ntype == "d":
                self.data["depots"].append(first)
            elif ntype == "s":
                self.data["satellites"].append(first)
            elif ntype == "f":
                self.data["stations"].append(first)
            elif ntype == "c":
                self.data["customers"].append(first)
            else:  # catch any exotic flag
                self.data.setdefault("others", []).append(first)

            idx += 1

        # -----------------------------------------------------------------
        # 3. global scalar parameters (remaining lines)
        # -----------------------------------------------------------------
        for line in lines[idx:]:
            # example  "L Large vehicle loading capacity /800.0/"
            key, *rest = line.split(None, 1)
            # value is always the last token between two '/' characters
            number_str = rest[0].split("/")[-2] if "/" in rest[0] else rest[-1].split("/")[-2]
            self.data["params"][key] = self._to_number(number_str)

    # ---------------------------------------------------------------------
    # tiny helper that converts strings to int or float whenever possible
    # ---------------------------------------------------------------------
    @staticmethod
    def _to_number(token: str) -> int | float | str:
        try:
            val = float(token)
            # convert to int if it’s a whole number (e.g., "30" -> 30)
            return int(val) if val.is_integer() else val
        except ValueError:
            return token
