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


def emit_header() -> str:
    return ".version 8.0\n.target sm_89\n.address_size 64\n"


def emit_signature(name: str, arg_order: list[str]) -> str:
    parameters = ",\n".join(
        f"    .param .u64 param_{argument}" for argument in arg_order
    )
    return f".visible .entry {name}(\n{parameters}\n)"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ir_path", type=Path, help="Path to a Kernel IR JSON file")
    args = parser.parse_args()
    graph = load_graph(args.ir_path)
    meta_path = args.ir_path.parent / "meta.json"
    with meta_path.open() as source:
        metadata = json.load(source)
    print(emit_header())
    for node in graph.nodes.values():
        if isinstance(node.op, KernelOp):
            kernel = node.op
            kernel_meta = next(
                entry for entry in metadata["kernels"]
                if entry["entry"] == kernel.name
            )
            print(emit_signature(kernel.name, kernel_meta["arg_order"]))
            print(f"Parameter order: {kernel_meta['arg_order']}", file=sys.stderr)
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
