"""Translate Emmy Kernel IR into PTX (implementation in progress)."""

import argparse
import json
import sys
from pathlib import Path

from emmy.compiler.graph import Graph
from emmy.compiler.ir.expr import BinaryExpr, Literal, Var
from emmy.compiler.ir.kernel.ir import KernelOp, Smem, StridedLoop, Sync, Tile, TreeHalve, WarpShuffle
from emmy.compiler.ir.loop import Accum, Assign, Cond, Load, Write


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
        self.shared_buffers: dict[str, tuple[int, ...]] = {}
        self.buffer_shapes: dict[str, tuple[int, ...]] = {}
        self.label_count = 0

    def emit(self, instruction: str) -> None:
        self.instructions.append(instruction)

    def new_label(self, prefix: str) -> str:
        label = f"{prefix}_{self.label_count}"
        self.label_count += 1
        return label

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


def emit_flat_index(buffer: str, indices, context: KernelContext) -> str:
    shape = (
        context.shared_buffers[buffer]
        if buffer in context.shared_buffers
        else context.buffer_shapes[buffer]
    )
    if len(indices) != len(shape):
        raise NotImplementedError("Index rank must match buffer rank")
    if not indices:
        return "0"

    offset = emit_index(indices[0], context)
    for extent, coordinate in zip(shape[1:], indices[1:]):
        value = emit_index(coordinate, context)
        result = context.new_register("u32")
        context.emit(f"mad.lo.u32 {result}, {offset}, {extent}, {value};")
        offset = result
    return offset


def emit_address(
    buffer: str, index: str, element_bytes: int, context: KernelContext
) -> str:
    offset = context.new_register("u64")
    address = context.new_register("u64")
    context.emit(f"mul.wide.u32 {offset}, {index}, {element_bytes};")
    context.emit(f"add.u64 {address}, %ptr_{buffer}, {offset};")
    return address


def emit_shared_address(
    buffer: str, index: str, context: KernelContext
) -> str:
    base = context.new_register("u32")
    offset = context.new_register("u32")
    address = context.new_register("u32")
    context.emit(f"mov.u32 {base}, {buffer};")
    context.emit(f"mul.lo.u32 {offset}, {index}, 4;")
    context.emit(f"add.u32 {address}, {base}, {offset};")
    return address


def emit_load(statement: Load, context: KernelContext) -> None:
    if (
        len(statement.names) != 1
        or statement.dtype is None
        or statement.dtype.name != "f32"
    ):
        raise NotImplementedError("Only scalar f32 global loads are supported")

    index = emit_flat_index(statement.input, statement.index, context)
    address = emit_address(
        statement.input, index, statement.dtype.nbytes, context
    )
    result = context.allocate_value(statement.names[0], "f32")
    context.emit(f"ld.global.f32 {result}, [{address}];")


def emit_assign(statement: Assign, context: KernelContext) -> None:
    if (
        statement.op.name == "relu"
        and statement.dtype is not None
        and statement.dtype.name == "f32"
        and len(statement.args) == 1
    ):
        source = context.value_registers[statement.args[0]]
        result = context.allocate_value(statement.name, "f32")
        context.emit(f"max.f32 {result}, {source}, 0f00000000;")
        return

    if (
        statement.op.name not in ("add", "multiply", "divide")
        or statement.dtype is None
        or statement.dtype.name != "f32"
        or len(statement.args) != 2
    ):
        raise NotImplementedError("Only binary f32 add/multiply/divide assignments are supported")

    left = context.value_registers[statement.args[0]]
    right = context.value_registers[statement.args[1]]
    result = context.allocate_value(statement.name, "f32")
    instruction = {
        "add": "add.f32",
        "multiply": "mul.f32",
        "divide": "div.rn.f32",
    }[statement.op.name]
    context.emit(f"{instruction} {result}, {left}, {right};")


def emit_write(statement: Write, context: KernelContext) -> None:
    if (
        statement.atomic
        or statement.swizzle != "NONE"
        or len(statement.index) not in (0, 1)
        or len(statement.values) not in (1, 2, 4)
        or statement.value_dtype is None
        or statement.value_dtype.name != "f32"
    ):
        raise NotImplementedError("Only plain scalar/1-D f32 writes of 1, 2, or 4 values are supported")

    index = (
        emit_index(statement.index[0], context)
        if statement.index
        else "0"
    )
    if statement.output in context.shared_buffers:
        address = emit_shared_address(statement.output, index, context)
        space = "shared"
    else:
        address = emit_address(
            statement.output, index, statement.value_dtype.nbytes, context
        )
        space = "global"
    values = [context.value_registers[name] for name in statement.values]
    if len(values) == 1:
        context.emit(f"st.{space}.f32 [{address}], {values[0]};")
    else:
        width = len(values)
        operands = ", ".join(values)
        context.emit(f"st.{space}.v{width}.f32 [{address}], {{{operands}}};")


def emit_strided_loop(statement: StridedLoop, context: KernelContext) -> None:
    if not statement.seed:
        raise NotImplementedError("Unseeded strided loops are not supported yet")
    if (
        not isinstance(statement.step, Literal)
        or statement.step.dtype != "int"
        or statement.step.value <= 0
    ):
        raise NotImplementedError("Only positive integer literal loop steps are supported")
    accumulators = [
        child for child in statement.body if isinstance(child, Accum)
    ]
    for accumulator in accumulators:
        if (
            accumulator.op.name != "add"
            or accumulator.dtype is None
            or accumulator.dtype.name != "f32"
            or accumulator.base is not None
        ):
            raise NotImplementedError("Only zero-based f32 sum accumulators are supported")
        register = context.allocate_value(accumulator.name, "f32")
        context.emit(f"mov.f32 {register}, 0f00000000;")
    start = emit_index(statement.start, context)
    index = context.new_register("u32")
    context.emit(f"mov.u32 {index}, {start};")
    previous_axis = context.axis_registers.get(statement.axis.name)
    context.axis_registers[statement.axis.name] = index
    loop_start = context.new_label("LOOP_START")
    loop_end = context.new_label("LOOP_END")
    done = context.new_register("pred")
    end = (
        emit_index(statement.end, context)
        if statement.end is not None
        else str(statement.axis.extent.value)
    )
    step = emit_index(statement.step, context)
    context.emit(f"{loop_start}:")
    context.emit(f"setp.ge.u32 {done}, {index}, {end};")
    context.emit(f"@{done} bra {loop_end};")
    for child in statement.body:
        emit_statement(child, context)
    context.emit(f"add.u32 {index}, {index}, {step};")
    context.emit(f"bra {loop_start};")
    context.emit(f"{loop_end}:")
    if previous_axis is None:
        del context.axis_registers[statement.axis.name]
    else:
        context.axis_registers[statement.axis.name] = previous_axis


def emit_accum(statement: Accum, context: KernelContext) -> None:
    if (
        statement.op.name != "add"
        or statement.dtype is None
        or statement.dtype.name != "f32"
        or statement.base is not None
    ):
        raise NotImplementedError("Only zero-based f32 sum accumulators are supported")

    accumulator = context.value_registers[statement.name]
    value = context.value_registers[statement.value]
    context.emit(f"add.f32 {accumulator}, {accumulator}, {value};")


def emit_warp_shuffle(statement: WarpShuffle, context: KernelContext) -> None:
    if (
        statement.dtype is None
        or statement.dtype.name != "f32"
        or statement.length not in (1, 2, 4, 8, 16, 32)
        or len(statement.state) != 1
        or len(statement.state_b) != 1
        or len(statement.combine_states) != 1
    ):
        raise NotImplementedError("Only single-state f32 warp sums are supported")

    combine = statement.combine_states[0]
    if not (
        isinstance(combine, Assign)
        and combine.op.name == "add"
        and combine.name == statement.state[0]
        and combine.args == (statement.state[0], statement.state_b[0])
    ):
        raise NotImplementedError("Only additive warp combines are supported")

    accumulator = context.value_registers[statement.state[0]]
    send_bits = context.new_register("b32")
    received_bits = context.new_register("b32")
    partner = context.new_register("f32")
    offset = statement.length // 2
    while offset > 0:
        context.emit(f"mov.b32 {send_bits}, {accumulator};")
        context.emit(
            f"shfl.sync.bfly.b32 {received_bits}, {send_bits}, "
            f"{offset}, 0x1f, 0xffffffff;"
        )
        context.emit(f"mov.b32 {partner}, {received_bits};")
        context.emit(f"add.f32 {accumulator}, {accumulator}, {partner};")
        offset //= 2


def emit_smem(statement: Smem, context: KernelContext) -> None:
    if statement.dtype != "float":
        raise NotImplementedError("Only f32 shared buffers are supported")

    elements = 1
    for extent in statement.extents:
        elements *= extent

    alignment = statement.align or 4
    context.declare(
        f".shared .align {alignment} .b8 {statement.name}[{elements * 4}];"
    )
    context.shared_buffers[statement.name] = tuple(statement.extents)


def emit_warp_coordinates(context: KernelContext) -> None:
    lane = context.new_register("u32")
    warp = context.new_register("u32")
    context.emit(f"and.b32 {lane}, %thread_linear, 31;")
    context.emit(f"shr.u32 {warp}, %thread_linear, 5;")
    context.axis_registers["lane"] = lane
    context.axis_registers["warp"] = warp


def emit_predicate(expression, context: KernelContext) -> str:
    if isinstance(expression, BinaryExpr) and expression.op == "==":
        left = emit_index(expression.left, context)
        right = emit_index(expression.right, context)
        predicate = context.new_register("pred")
        context.emit(f"setp.eq.u32 {predicate}, {left}, {right};")
        return predicate
    raise NotImplementedError(
        f"Unsupported predicate: {type(expression).__name__}"
    )


def emit_cond(statement: Cond, context: KernelContext) -> None:
    if statement.else_body:
        raise NotImplementedError("Conditional else branches are not supported yet")

    predicate = emit_predicate(statement.cond, context)
    end = context.new_label("COND_END")
    context.emit(f"@!{predicate} bra {end};")
    for child in statement.body:
        emit_statement(child, context)
    context.emit(f"{end}:")


def emit_sync(statement: Sync, context: KernelContext) -> None:
    if statement.warp or statement.count is not None:
        raise NotImplementedError("Only full-block barriers are supported")
    context.emit(f"bar.sync {statement.barrier_id};")


def emit_tree_halve(statement: TreeHalve, context: KernelContext) -> None:
    length = statement.length
    if (
        length < 1
        or length & (length - 1)
        or statement.dtype is None
        or statement.dtype.name != "f32"
        or len(statement.bufs) != 1
        or len(statement.state) != 1
        or len(statement.state_b) != 1
        or len(statement.combine_states) != 1
        or statement.barrier_count is not None
        or statement.inner is not None
    ):
        raise NotImplementedError("Only single-state f32 block tree sums are supported")

    combine = statement.combine_states[0]
    if not (
        isinstance(combine, Assign)
        and combine.op.name == "add"
        and combine.name == statement.state[0]
        and combine.args == (statement.state[0], statement.state_b[0])
    ):
        raise NotImplementedError("Only additive tree combines are supported")

    buffer = statement.bufs[0]
    if context.shared_buffers.get(buffer) != (statement.length,):
        raise NotImplementedError("Tree sum requires a matching 1-D shared buffer")

    tid = context.axis_registers[statement.tid_var]
    accumulator = context.value_registers[statement.state[0]]
    partner = context.new_register("f32")
    if statement.tid_var != "warp":
        raise NotImplementedError("Only warp-indexed tree sums are supported")

    lane = context.axis_registers["lane"]
    skip = context.new_register("pred")
    stride = statement.length // 2
    while stride > 0:
        stage_end = context.new_label("TREE_STAGE_END")
        context.emit(f"setp.ne.u32 {skip}, {lane}, 0;")
        context.emit(f"@{skip} bra {stage_end};")
        context.emit(f"setp.ge.u32 {skip}, {tid}, {stride};")
        context.emit(f"@{skip} bra {stage_end};")

        partner_index = context.new_register("u32")
        context.emit(f"add.u32 {partner_index}, {tid}, {stride};")
        left_address = emit_shared_address(buffer, tid, context)
        right_address = emit_shared_address(buffer, partner_index, context)

        context.emit(f"ld.shared.f32 {accumulator}, [{left_address}];")
        context.emit(f"ld.shared.f32 {partner}, [{right_address}];")
        context.emit(f"add.f32 {accumulator}, {accumulator}, {partner};")
        context.emit(f"st.shared.f32 [{left_address}], {accumulator};")

        context.emit(f"{stage_end}:")
        context.emit(f"bar.sync {statement.barrier_id};")
        stride //= 2

    result_address = emit_shared_address(buffer, "0", context)
    context.emit(f"ld.shared.f32 {accumulator}, [{result_address}];")


def emit_statement(statement, context: KernelContext) -> None:
    if isinstance(statement, TreeHalve):
        emit_tree_halve(statement, context)
        return
    if isinstance(statement, Sync):
        emit_sync(statement, context)
        return
    if isinstance(statement, Cond):
        emit_cond(statement, context)
        return
    if isinstance(statement, Smem):
        emit_smem(statement, context)
        return
    if isinstance(statement, WarpShuffle):
        emit_warp_shuffle(statement, context)
        return
    if isinstance(statement, StridedLoop):
        emit_strided_loop(statement, context)
        return
    if isinstance(statement, Accum):
        emit_accum(statement, context)
        return
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
            for graph_node in graph.nodes.values():
                for tensor in graph_node.outputs:
                    context.buffer_shapes[tensor.name] = tuple(
                        dimension.value for dimension in tensor.shape
                    )
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
            emit_warp_coordinates(context)
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
