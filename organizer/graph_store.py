from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable

from corpusgen.graph_records import GraphAddress, GraphRow


class AtomicGraphStore:
    def __init__(self, rows: Iterable[GraphRow] = ()) -> None:
        self._rows: dict[GraphAddress, GraphRow] = {}
        self.hits = 0
        self.misses = 0
        for row in rows:
            self.add(row)

    def add(self, row: GraphRow) -> None:
        if row.address in self._rows:
            raise ValueError(f"duplicate graph address: {row.address}")
        self._rows[row.address] = row

    def lookup(self, address: GraphAddress) -> GraphRow | None:
        row = self._rows.get(address)
        if row is None:
            self.misses += 1
        else:
            self.hits += 1
        return row

    def reset_counters(self) -> None:
        self.hits = 0
        self.misses = 0

    def rows(self) -> tuple[GraphRow, ...]:
        return tuple(self._rows[key] for key in sorted(self._rows))

    def canonical_bytes(self) -> bytes:
        lines = [
            json.dumps(row.as_json(), sort_keys=True, separators=(",", ":"))
            for row in self.rows()
        ]
        return ("\n".join(lines) + ("\n" if lines else "")).encode()

    def snapshot_sha256(self) -> str:
        return hashlib.sha256(self.canonical_bytes()).hexdigest()

    def save(self, path: str | Path) -> None:
        Path(path).write_bytes(self.canonical_bytes())

    @classmethod
    def load(cls, path: str | Path) -> "AtomicGraphStore":
        rows = [
            GraphRow.from_json(json.loads(line))
            for line in Path(path).read_text().splitlines()
            if line
        ]
        return cls(rows)

    def __len__(self) -> int:
        return len(self._rows)
