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
            print("{")
            for argument in kernel_meta["arg_order"]:
                print(f"    .reg .b64 %ptr_{argument};")
            for axis in ("x", "y", "z"):
                print(f"    .reg .b32 %thread_{axis};")
            for axis in ("x", "y", "z"):
                print(f"    .reg .b32 %block_{axis};")
            for axis in ("x", "y", "z"):
                print(f"    .reg .b32 %block_size_{axis};")
            print("    .reg .b32 %thread_linear;")
            print("    .reg .b32 %block_linear;")
            print("    .reg .b32 %threads_per_block;")
            print("    .reg .b32 %global_linear;")
            for axis in ("x", "y", "z"):
                print(f"    .reg .b32 %grid_size_{axis};")
            for argument in kernel_meta["arg_order"]:
                print(f"    ld.param.u64 %ptr_{argument}, [param_{argument}];")
            for argument in kernel_meta["arg_order"]:
                print(f"    cvta.to.global.u64 %ptr_{argument}, %ptr_{argument};")
            for axis in ("x", "y", "z"):
                print(f"    mov.u32 %thread_{axis}, %tid.{axis};")
            for axis in ("x", "y", "z"):
                print(f"    mov.u32 %block_{axis}, %ctaid.{axis};")
            for axis in ("x", "y", "z"):
                print(f"    mov.u32 %block_size_{axis}, %ntid.{axis};")
            for axis in ("x", "y", "z"):
                print(f"    mov.u32 %grid_size_{axis}, %nctaid.{axis};")
            print("    mad.lo.u32 %thread_linear, %block_size_y, %thread_z, %thread_y;")
            print("    mad.lo.u32 %thread_linear, %block_size_x, %thread_linear, %thread_x;")
            print("    mad.lo.u32 %block_linear, %grid_size_y, %block_z, %block_y;")
            print("    mad.lo.u32 %block_linear, %grid_size_x, %block_linear, %block_x;")
            print("    mul.lo.u32 %threads_per_block, %block_size_x, %block_size_y;")
            print("    mul.lo.u32 %threads_per_block, %threads_per_block, %block_size_z;")
            print("    mad.lo.u32 %global_linear, %block_linear, %threads_per_block, %thread_linear;")
            print("    ret;")
            print("}")
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
