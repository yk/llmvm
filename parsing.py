import re
from loguru import logger
from interface import Arg, Load, Icmp, Srem, Add, Mul, Call, Assign, Store, Branch, BranchCond, Return, Program, to_vtype, GetElementPtr, Copy, Switch, AllocArray, Alloc

def _line_stripper(in_f):
    for line in in_f:
        line = line.rstrip()
        if not line:
            continue
        yield line

def parse_arg(arg):
    logger.debug(f"parse_arg({arg})")
    if m := re.match(r"ptr noundef (\S+)", arg):
        return Arg(vtype="str", value=m.group(1))
    if m := re.match(r"i32 noundef (\S+)", arg):
        return Arg(vtype="i32", value=m.group(1))
    raise NotImplementedError(arg)

def parse_call(expr):
    logger.debug(f"parse_call({expr})")
    if m := re.match(r"\s*call \w+(?: \(.*\))? @(\w+)\((.*)\)", expr):
        name, args = m.groups()
        args = args.split(", ")
        args = [parse_arg(arg) for arg in args if arg]
        return Call(name=name, args=args)
    return None

def parse_expr(expr):
    if m := re.match(r"alloca \[(\d+) x (\S+)\]", expr):
        size, vtype = m.groups()
        return AllocArray(vtype=vtype, size=int(size))
    if m := re.match(r"alloca (\S+),", expr):
        vtype = m.group(1)
        return Alloc(vtype=vtype)
    if m := re.match(r"sext \S+ (\S+) to \S+", expr):
        return Copy(ptr=m.group(1))
    if m := re.match(r"load (\w+), ptr (%\d+),", expr):
        return Load(vtype=m.group(1), ptr=m.group(2))
    if m := re.match(r"icmp (eq|ne|sgt|sge|slt|sle) (\w+) (\S+), (\S+)", expr):
        op, vtype, lhs, rhs = m.groups()
        return Icmp(vtype=vtype, op=op, lhs=lhs, rhs=rhs)
    if m := re.match(r"srem (\w+) (\S+), (\S+)", expr):
        vtype, lhs, rhs = m.groups()
        return Srem(vtype=vtype, lhs=lhs, rhs=rhs)
    if m := re.match(r"add nsw (\w+) (\S+), (\S+)", expr):
        vtype, lhs, rhs = m.groups()
        return Add(vtype=vtype, lhs=lhs, rhs=rhs)
    if m := re.match(r"mul nsw (\w+) (\S+), (\S+)", expr):
        vtype, lhs, rhs = m.groups()
        return Mul(vtype=vtype, lhs=lhs, rhs=rhs)
    if call := parse_call(expr):
        if call is not None:
            logger.debug(f"found call {call}")
            return call
    if m := re.match(r"getelementptr inbounds \[\d+ x (\S+)\], ptr (\S+), i32 0, i32 (\S+)", expr):
        vtype, ptr, idx = m.groups()
        return GetElementPtr(vtype=vtype, ptr=ptr, idx=idx)
    
    raise NotImplementedError(expr)


def parse_switch(in_f):
    cases = {}
    for line in _line_stripper(in_f):
        if re.fullmatch(r"\s+\]", line):
            break
        if m := re.match(r"\s+i32 (\S+), label %(\d+)", line):
            value, label = m.groups()
            cases[value] = label
            continue
        raise NotImplementedError(line)
    else:
        raise ValueError("Expected ']' in switch")
    return cases


def parse_instructions(in_f):
    instructions = []
    labels = {}
    for line in _line_stripper(in_f):
        if re.fullmatch(r"\}", line):
            break
        if m := re.match(r"(\d+):", line):
            label = m.group(1)
            labels[label] = len(instructions)
            continue
        if m := re.fullmatch(r"\s+(%\d+) = (.*)", line): # register assignment
            reg, expr = m.groups()
            expr = parse_expr(expr)
            if expr is not None:
                instructions.append(Assign(reg=reg, expr=expr))
            continue
        if m := re.match(r"\s+store (\w+) (\S+), ptr (\S+),", line):
            vtype, value, ptr = m.groups()
            instructions.append(Store(vtype=vtype, value=value, ptr=ptr))
            continue
        if m := re.match(r"\s+br label %(\d+)", line):
            label = m.group(1)
            instructions.append(Branch(label=label))
            continue
        if m := re.match(r"\s+br i1 (%\d+), label %(\d+), label %(\d+)", line):
            cond_reg, label_true, label_false = m.groups()
            instructions.append(BranchCond(cond_reg=cond_reg, label_true=label_true, label_false=label_false))
            continue
        if m := re.match(r"\s+ret (\S+) (\S+)", line):
            vtype, value = m.groups()
            instructions.append(Return(vtype=vtype, value=value))
            continue
        if call := parse_call(line):
            if call is not None:
                logger.debug(f"found call {call}")
                instructions.append(call)
                continue
        if m := re.match(r"\s+switch \S+ (\S+), label %(\d+) \[", line):
            ptr, default_label = m.groups()
            cases = parse_switch(in_f)
            instructions.append(Switch(ptr=ptr, default_label=default_label, cases=cases))
            continue
        
        raise NotImplementedError(line)
    
    return instructions, labels


def parse_program(in_f):
    constants = {}
    for line in _line_stripper(in_f):
        if m := re.match(r'(@\.str(?:\.\d+)?) .* c"([^"]+)\\00",', line):
            name, value = m.groups()
            value = value.replace(r"\0A", "\n")
            constants[name] = to_vtype(value=value, vtype="str")
            continue
        if re.fullmatch(r"define .* \{", line):
            instructions, labels = parse_instructions(in_f)
            break
        if line.split()[0] in (";", "target", "source_filename"):
            continue
        raise NotImplementedError(line)
    else:
        raise NotImplementedError()
    program = Program(instructions=instructions, labels=labels, constants=constants)
    logger.debug(f"parsed program: {program}")
    logger.debug("Instructions:")
    for instr in program.instructions:
        logger.debug(f"  {instr}")
    
    return program
