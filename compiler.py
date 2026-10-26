"""Translate Emmy Kernel IR into PTX (implementation in progress)."""

import argparse
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Tile
from emmy.compiler.ir.loop import Assign, Load, Write


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


def axis_strides(tile: Tile) -> list[int]:
    strides = []
    stride = 1
    for axis in reversed(tile.axes):
        strides.append(stride)
        stride *= axis.extent.value
    return list(reversed(strides))


def tile_size(tile: Tile) -> int:
    size = 1
    for axis in tile.axes:
        size *= axis.extent.value
    return size


class KernelContext:
    def __init__(self) -> None:
        self.declarations: list[str] = []
        self.instructions: list[str] = []
        self.value_registers: dict[str, str] = {}
        self.axis_registers: dict[str, str] = {}

    def emit(self, instruction: str) -> None:
        self.instructions.append(instruction)

    def declare(self, declaration: str) -> None:
        self.declarations.append(declaration)

    def new_register(self, ptx_type: str) -> str:
        register = f"%temp_{len(self.declarations)}"
        self.declare(f".reg .{ptx_type} {register};")
        return register

    def allocate_value(self, name: str, ptx_type: str) -> str:
        if name not in self.value_registers:
            self.value_registers[name] = self.new_register(ptx_type)
        return self.value_registers[name]

    def render_body(self) -> str:
        lines = self.declarations + self.instructions
        body = "\n".join(f"    {line}" for line in lines)
        return "{\n" + body + "\n}"


def emit_index(expression, context: KernelContext) -> str:
    if isinstance(expression, Var):
        return context.axis_registers[expression.name]
    if isinstance(expression, Literal) and expression.dtype == "int":
        return str(expression.value)
    if isinstance(expression, BinaryExpr) and expression.op in ("+", "*"):
        left = emit_index(expression.left, context)
        right = emit_index(expression.right, context)
        result = context.new_register("u32")
        instruction = "add.u32" if expression.op == "+" else "mul.lo.u32"
        context.emit(f"{instruction} {result}, {left}, {right};")
        return result
    raise NotImplementedError(
        f"Unsupported index expression: {type(expression).__name__}"
    )


def emit_address(
    buffer: str, index: str, element_bytes: int, context: KernelContext
) -> str:
    offset = context.new_register("u64")
    address = context.new_register("u64")
    context.emit(f"mul.wide.u32 {offset}, {index}, {element_bytes};")
    context.emit(f"add.u64 {address}, %ptr_{buffer}, {offset};")
    return address


def emit_load(statement: Load, context: KernelContext) -> None:
    if (
        len(statement.names) != 1
        or len(statement.index) != 1
        or statement.dtype is None
        or statement.dtype.name != "f32"
    ):
        raise NotImplementedError("Only scalar 1-D f32 loads are supported")

    index = emit_index(statement.index[0], context)
    address = emit_address(
        statement.input, index, statement.dtype.nbytes, context
    )
    result = context.allocate_value(statement.names[0], "f32")
    context.emit(f"ld.global.f32 {result}, [{address}];")


def emit_assign(statement: Assign, context: KernelContext) -> None:
    if (
        statement.op.name != "add"
        or statement.dtype is None
        or statement.dtype.name != "f32"
        or len(statement.args) != 2
    ):
        raise NotImplementedError("Only binary f32 add assignments are supported")

    left = context.value_registers[statement.args[0]]
    right = context.value_registers[statement.args[1]]
    result = context.allocate_value(statement.name, "f32")
    context.emit(f"add.f32 {result}, {left}, {right};")


def emit_write(statement: Write, context: KernelContext) -> None:
    if (
        statement.atomic
        or statement.swizzle != "NONE"
        or len(statement.index) != 1
        or len(statement.values) not in (1, 2)
        or statement.value_dtype is None
        or statement.value_dtype.name != "f32"
    ):
        raise NotImplementedError("Only plain 1-D f32 scalar/pair writes are supported")

    index = emit_index(statement.index[0], context)
    address = emit_address(
        statement.output, index, statement.value_dtype.nbytes, context
    )
    values = [context.value_registers[name] for name in statement.values]
    if len(values) == 1:
        context.emit(f"st.global.f32 [{address}], {values[0]};")
    else:
        operands = ", ".join(values)
        context.emit(f"st.global.v2.f32 [{address}], {{{operands}}};")


def emit_statement(statement, context: KernelContext) -> None:
    if isinstance(statement, Load):
        emit_load(statement, context)
        return
    if isinstance(statement, Assign):
        emit_assign(statement, context)
        return
    if isinstance(statement, Write):
        emit_write(statement, context)
        return
    raise NotImplementedError(
        f"Unsupported statement: {type(statement).__name__}"
    )


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
            context = KernelContext()
            kernel_meta = next(
                entry for entry in metadata["kernels"]
                if entry["entry"] == kernel.name
            )
            print(emit_signature(kernel.name, kernel_meta["arg_order"]))
            for argument in kernel_meta["arg_order"]:
                context.declare(f".reg .b64 %ptr_{argument};")
            for axis in ("x", "y", "z"):
                context.declare(f".reg .b32 %thread_{axis};")
            for axis in ("x", "y", "z"):
                context.declare(f".reg .b32 %block_{axis};")
            for axis in ("x", "y", "z"):
                context.declare(f".reg .b32 %block_size_{axis};")
            context.declare(".reg .b32 %thread_linear;")
            context.declare(".reg .b32 %block_linear;")
            context.declare(".reg .b32 %threads_per_block;")
            context.declare(".reg .b32 %global_linear;")
            for axis in ("x", "y", "z"):
                context.declare(f".reg .b32 %grid_size_{axis};")
            for statement in kernel.body:
                if isinstance(statement, Tile):
                    context.declare(".reg .pred %outside_tile;")
                    for axis in statement.axes:
                        register = f"%axis_{axis.name}"
                        context.declare(f".reg .b32 {register};")
                        context.axis_registers[axis.name] = register
            for argument in kernel_meta["arg_order"]:
                context.emit(f"ld.param.u64 %ptr_{argument}, [param_{argument}];")
            for argument in kernel_meta["arg_order"]:
                context.emit(f"cvta.to.global.u64 %ptr_{argument}, %ptr_{argument};")
            for axis in ("x", "y", "z"):
                context.emit(f"mov.u32 %thread_{axis}, %tid.{axis};")
            for axis in ("x", "y", "z"):
                context.emit(f"mov.u32 %block_{axis}, %ctaid.{axis};")
            for axis in ("x", "y", "z"):
                context.emit(f"mov.u32 %block_size_{axis}, %ntid.{axis};")
            for axis in ("x", "y", "z"):
                context.emit(f"mov.u32 %grid_size_{axis}, %nctaid.{axis};")
            context.emit("mad.lo.u32 %thread_linear, %block_size_y, %thread_z, %thread_y;")
            context.emit("mad.lo.u32 %thread_linear, %block_size_x, %thread_linear, %thread_x;")
            context.emit("mad.lo.u32 %block_linear, %grid_size_y, %block_z, %block_y;")
            context.emit("mad.lo.u32 %block_linear, %grid_size_x, %block_linear, %block_x;")
            context.emit("mul.lo.u32 %threads_per_block, %block_size_x, %block_size_y;")
            context.emit("mul.lo.u32 %threads_per_block, %threads_per_block, %block_size_z;")
            context.emit("mad.lo.u32 %global_linear, %block_linear, %threads_per_block, %thread_linear;")
            for statement in kernel.body:
                if isinstance(statement, Tile):
                    context.emit(f"setp.ge.u32 %outside_tile, %global_linear, {tile_size(statement)};")
                    context.emit("@%outside_tile bra TILE_END;")
                    for axis, stride in zip(statement.axes, axis_strides(statement)):
                        context.emit(f"div.u32 %axis_{axis.name}, %global_linear, {stride};")
                        context.emit(f"rem.u32 %axis_{axis.name}, %axis_{axis.name}, {axis.extent.value};")
                    for child in statement.body:
                        emit_statement(child, context)
                    context.emit("TILE_END:")
            context.emit("ret;")
            print(context.render_body())
            print(f"Parameter order: {kernel_meta['arg_order']}", file=sys.stderr)
            print(f"Inputs: {node.inputs}", file=sys.stderr)
            for output in node.outputs:
                print(
                    f"Output: {output.name}, shape={output.shape}, dtype={output.dtype}",
                    file=sys.stderr,
                )
            for statement in kernel.body:
                print(type(statement).__name__, file=sys.stderr)
                if isinstance(statement, Tile):
                    for axis, stride in zip(statement.axes, axis_strides(statement)):
                        print(
                            f"Axis: {axis.name}, extent={axis.extent}, stride={stride}",
                            file=sys.stderr,
                        )
            print(kernel.name, file=sys.stderr)
    print(f"Loaded graph with {len(graph.nodes)} nodes", file=sys.stderr)


if __name__ == "__main__":
    main()
