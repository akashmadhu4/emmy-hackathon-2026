"""Translate Emmy Kernel IR into PTX (implementation in progress)."""

import argparse
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.kernel.ir import KernelOp


def load_graph(path: Path) -> Graph:
    """Deserialize the JSON graph into Emmy's typed IR objects."""
    with path.open() as source:
        return Graph.from_dict(json.load(source))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_path", type=Path, help="Path to a Kernel IR JSON file")
    args = parser.parse_args()
    graph = load_graph(args.ir_path)
    for node in graph.nodes.values():
        if isinstance(node.op, KernelOp):
            kernel = node.op
            print(f"Inputs: {node.inputs}", file=sys.stderr)
            for output in node.outputs:
                print(
                    f"Output: {output.name}, shape={output.shape}, dtype={output.dtype}",
                    file=sys.stderr,
                )
            for statement in kernel.body:
                print(type(statement).__name__, file=sys.stderr)
            print(kernel.name, file=sys.stderr)
    print(f"Loaded graph with {len(graph.nodes)} nodes", file=sys.stderr)


if __name__ == "__main__":
    main()
