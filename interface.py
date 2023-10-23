from typing import Any  

import pydantic

def to_vtype(value, vtype):
    match vtype:
        case "i32":
            return int(value)
        case "i8":
            return str(value)
        case "str":
            return str(value)
    raise NotImplementedError(vtype)


class Instruction(pydantic.BaseModel):
    kind: str

class Expr(Instruction):
    pass

class Load(Expr):
    kind: str = "load"
    vtype: str
    ptr: str

class Copy(Expr):
    kind: str = "copy"
    ptr: str

class Alloc(Expr):
    kind: str = "alloc"
    vtype: str

class AllocArray(Expr):
    kind: str = "alloc_array"
    vtype: str
    size: int

class GetElementPtr(Expr):
    kind: str = "get_element_ptr"
    vtype: str
    ptr: str
    idx: str

class Icmp(Expr):
    kind: str = "icmp"
    vtype: str
    op: str
    lhs: str
    rhs: str

class Srem(Expr):
    kind: str = "srem"
    vtype: str
    lhs: str
    rhs: str

class Add(Expr):
    kind: str = "add"
    vtype: str
    lhs: str
    rhs: str

class Mul(Expr):
    kind: str = "mul"
    vtype: str
    lhs: str
    rhs: str


class Arg(pydantic.BaseModel):
    vtype: str
    value: str

class Call(Expr):
    kind: str = "call"
    name: str
    args: list[Arg]


class Assign(Instruction):
    kind: str = "assign"
    reg: str
    expr: Expr

class Store(Instruction):
    kind: str = "store"
    vtype: str
    value: str
    ptr: str

class Branch(Instruction):
    kind: str = "branch"
    label: str


class BranchCond(Instruction):
    kind: str = "branch_cond"
    cond_reg: str
    label_true: str
    label_false: str

class Return(Instruction):
    kind: str = "return"
    vtype: str
    value: str

class Switch(Instruction):
    kind: str = "switch"
    ptr: str
    default_label: str
    cases: dict[str, str]
        
class Program(pydantic.BaseModel):
    instructions: list[Instruction]
    labels: dict[str, int]
    constants: dict[str, Any]