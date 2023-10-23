from typing import Any
import random
import re
from loguru import logger
import openai
import os

import pydantic

from interface import (
    Alloc,
    AllocArray,
    Expr,
    to_vtype,
    Load,
    Icmp,
    Srem,
    Copy,
    Add,
    Mul,
    Call,
    Assign,
    Store,
    Branch,
    BranchCond,
    Return,
    Program,
    GetElementPtr,
    Switch,
)


class VM(pydantic.BaseModel):
    name: str = "VM"
    constants: dict[str, Any]
    registers: dict[str, Any] = {}
    memory: dict[str, Any] = {}

    def _get_from_registers(self, value):
        return self.registers[value]
    
    def _to_vtype(self, value, vtype):
        return to_vtype(value, vtype)

    def get_value(self, value, vtype=None):
        if isinstance(value, str):
            if value.startswith("%"):
                value = self._get_from_registers(value)
                if isinstance(value, str):
                    logger.debug(f"found string {value}")
                    if m := re.match(r"arr ptr=(\S+) idx=(\S+)", value):
                        ptr, idx = m.groups()
                        logger.debug(f"performing array lookup on {ptr} {idx}")
                        arr = self._get_from_registers(ptr)
                        logger.debug(f"arr = {arr}")
                        value = self._load_array(arr, idx)
                        logger.debug(f"array lookup result = {value}")
            elif value.startswith("@"):
                value = self.constants[value]
        if vtype is not None:
            value = self._to_vtype(value, vtype)
        return value

    def _cmp_with_op(self, op, lhs, rhs) -> bool:
        match op:
            case "eq":
                return lhs == rhs
            case "ne":
                return lhs != rhs
            case "sgt":
                return lhs > rhs
            case "sge":
                return lhs >= rhs
            case "slt":
                return lhs < rhs
            case "sle":
                return lhs <= rhs
        raise NotImplementedError(op)


    def _rem(self, lhs, rhs):
        return lhs % rhs

    def _add(self, lhs, rhs):
        return lhs + rhs
    
    def _mul(self, lhs, rhs):
        return lhs * rhs
    
    def _load_array(self, arr, idx):
        return arr[idx]
    
    def _alloc_array(self, vtype, size):
        if vtype in ("i32", "i8"):
            return {str(i): 0 for i in range(size)}
        raise NotImplementedError(vtype)

    def eval_expr(self, expr: Expr):
        logger.debug(f"eval_expr({expr})")
        match expr:
            case Copy(ptr=ptr):
                return self.get_value(ptr)
            case Alloc(vtype=vtype):
                return self.get_value(0, vtype)
            case AllocArray(vtype=vtype, size=size):
                return self._alloc_array(vtype, size)
            case Load(vtype=vtype, ptr=ptr):
                return self._to_vtype(self._get_from_registers(ptr), vtype)
            case Icmp(vtype=vtype, op=op, lhs=lhs, rhs=rhs):
                logger.debug(f"icmp {vtype} {op} {lhs} {rhs}")
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._cmp_with_op(op, lhs, rhs)
            case Srem(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._rem(lhs, rhs)
            case Add(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._add(lhs, rhs)
            case Mul(vtype=vtype, lhs=lhs, rhs=rhs):
                lhs = self.get_value(lhs, vtype)
                rhs = self.get_value(rhs, vtype)
                return self._mul(lhs, rhs)
            case Call(name=name, args=args):
                logger.debug(f"calling {name} {args}")
                match name:
                    case "printf":
                        logger.debug(f"printf {args}")
                        args = [self.get_value(arg.value, arg.vtype) for arg in args]
                        def _maybe_to_char(value):
                            try:
                                return chr(int(value))
                            except ValueError:
                                return value
                        args = [_maybe_to_char(arg) for arg in args]
                        logger.debug(f"printf {args}")
                        fstring = args[0]
                        # we need to replace all the percent-formatters (like %d) with {}
                        fstring = re.sub(r"%\w", "{}", fstring)
                        out_str = fstring.format(*args[1:])
                        print(out_str, end="")
                        return 0
                    case "srand":
                        seed = self.get_value(args[0].value, args[0].vtype)
                        self._rng = random.Random(seed)
                        return
                    case "scanf":
                        logger.debug(f"scanf {args}")
                        dst = args[1].value
                        prompt = self.get_value(args[0].value, "str")
                        assert prompt.strip() == "%c"
                        input_str = input()
                        char = ord(input_str)
                        self.store(dst, char)
                        return 0
                    case "rand":
                        return self._rng.randint(0, 100)
                raise NotImplementedError(name)
            case GetElementPtr(vtype=vtype, ptr=ptr, idx=idx):
                logger.debug(f"getting element ptr from {ptr} -> {idx}")
                idx = self.get_value(idx)
                logger.debug(f"idx into array is {idx}")
                return f"arr ptr={ptr} idx={idx}"


        raise NotImplementedError(expr)

    def _store(self, ptr, value):
        self.registers[ptr] = value

    def _is_array_ref(self, value):
        return isinstance(value, str) and value.startswith("arr ptr")
    
    def _parse_array_ref(self, value):
        m = re.match(r"arr ptr=(\S+) idx=(\S+)", value)
        ptr, idx = m.groups()
        return ptr, idx
    
    def _store_array(self, ptr, idx, value):
        logger.debug(f"called store_array ptr={ptr} idx={idx} value={value}")
        arr = self._get_from_registers(ptr)
        logger.debug(f"before store: arr = {arr}")
        arr[idx] = value
        logger.debug(f"after store: arr = {arr}")
    
    def assign(self, reg, value):
        logger.debug(f"assign {value} to {reg}")
        self._store(reg, value)
    
    def store(self, ptr, value):
        logger.debug(f"store {value} to {ptr}")
        # check if ptr points to an array reference
        reg_value = self._get_from_registers(ptr)
        logger.debug(f"ptr {ptr} exists, value = {reg_value}")
        if self._truthy(self._is_array_ref(reg_value)):
            logger.debug(f"ptr {ptr} is an array ref")
            arr_ptr, idx = self._parse_array_ref(reg_value)
            logger.debug(f"parsed array ref: {arr_ptr} {idx}")
            self._store_array(arr_ptr, idx, value)
            return
        self._store(ptr, value)
    
    def _truthy(self, value):
        return bool(value)

    def run_program(self, program: Program):
        iptr = 0
        num_instr = 0

        while iptr < len(program.instructions):
            instr = program.instructions[iptr]
            logger.debug(f"{num_instr} || executing {instr}")
            num_instr += 1
            
            match instr:
                case Return(vtype=vtype, value=value):
                    logger.debug(f"return {value}")
                    return self.get_value(value, vtype=vtype)
                case Branch(label=label):
                    logger.debug(f"branch to {label}")
                    iptr = program.labels[label]
                    continue
                case BranchCond(cond_reg=cond_reg, label_true=label_true, label_false=label_false):
                    cond = self.get_value(cond_reg)
                    logger.debug(f"cond {cond_reg} = {cond} (type {type(cond)}))")
                    if self._truthy(cond):
                        logger.debug(f"branch to {label_true}")
                        iptr = program.labels[label_true]
                    else:
                        logger.debug(f"branch to {label_false}")
                        iptr = program.labels[label_false]
                    continue
                case Store(value=value, ptr=ptr):
                    logger.debug(f"store {value} to {ptr}")
                    value = self.get_value(value)
                    self.store(ptr, value)
                case Assign(reg=reg, expr=expr):
                    value = self.eval_expr(expr)
                    self.assign(reg, value)
                case Switch(ptr=ptr, default_label=default_label, cases=cases):
                    value = self.get_value(ptr)
                    logger.debug(f"switch on {value}")
                    for case, label in cases.items():
                        if self._truthy(self._cmp_with_op("eq", value, case)):
                            logger.debug(f"branch to case {case} -> {label}")
                            iptr = program.labels[cases[case]]
                            break
                    else:
                        logger.debug(f"branch to default {default_label}")
                        iptr = program.labels[default_label]
                    continue
                case _:
                    if isinstance(instr, Expr):
                        self.eval_expr(instr)
                    else:
                        raise NotImplementedError(instr)
                    

            iptr += 1

openai.api_key = os.getenv("OPENAI_KEY")


def gpt_get_answer(prompt, system=""):
    logger.debug(f"Getting answer for prompt: {prompt} | system: {system}")
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    
    prompt = f"""{prompt}
    Write your answer on a single line. Be very short and concise. I just want the answer, not the explanation."""

    messages.append({"role": "user", "content": prompt})
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=messages,
        max_tokens=1024,
    )
    response = response["choices"][0]["message"]["content"].strip()  # type: ignore
    logger.debug(f"Got answer: {response}")
    return response


class GPTVM(VM):
    name: str = "GPT VM"
    registers: dict[str, Any] = {}

    def _to_vtype(self, value, vtype):
        return value

    def _get_from_registers(self, value):
        prommpt = f"""Look at registers: {self.registers}.
        Now get me the value of {value}."""
        return gpt_get_answer(prommpt)
    
    def _add(self, lhs, rhs):
        prompt = f"""Add {lhs} and {rhs}."""
        return gpt_get_answer(prompt)
    
    def _rem(self, lhs, rhs):
        prompt = f"""Get the remainder value of {lhs} divided by {rhs}."""
        return gpt_get_answer(prompt)
    
    def _mul(self, lhs, rhs):
        prompt = f"""Multiply {lhs} and {rhs}."""
        return gpt_get_answer(prompt)
    
    def _load_array(self, arr, idx):
        prompt = f"""Get the value of index {idx} in array {arr}."""
        return gpt_get_answer(prompt)
    
    def _truthy(self, value):
        if isinstance(value, bool):
            b = value
        elif isinstance(value, int):
            b = bool(value)
        elif isinstance(value, str):
            if any(
                t in value.lower()
                for t in (
                    "yes", "true"
                )
            ):
                b = True
            else:
                b = False
        else:
            raise NotImplementedError(value)
        logger.debug(f"truthiness of {value} = {b}")
        return b
    
    def _cmp_with_op(self, op, lhs, rhs) -> bool:
        match op:
            case "eq":
                prompt = f"""Are {lhs} and {rhs} equal?"""
            case "ne":
                prompt = f"""Are {lhs} and {rhs} different?"""
            case "sgt":
                prompt = f"""Is {lhs} greater than {rhs}?"""
            case "sge":
                prompt = f"""Is {lhs} greater than or equal to {rhs}?"""
            case "slt":
                prompt = f"""Is {lhs} less than {rhs}?"""
            case "sle":
                prompt = f"""Is {lhs} less than or equal to {rhs}?"""
            case _:
                raise NotImplementedError(op)
        return gpt_get_answer(
            system="You are a very keen observer.",
            prompt=prompt)
    
    def _store(self, ptr, value):
        self.registers[ptr] = value

    def _is_array_ref(self, value):
        prompt = f"""Is {value} an array reference?"""
        return gpt_get_answer(prompt)
    
    def _parse_array_ref(self, value):
        prompt = f"""Parse array reference {value}. Give me the array pointer and the index separated by a pipe symbol (|)"""
        return gpt_get_answer(prompt).split("|")
    
    def _store_array(self, ptr, idx, value):
        arr = self._get_from_registers(ptr)
        prompt = f"""Store {value} in array {arr} at index {idx}. Give me back the new array."""
        answer = gpt_get_answer(prompt)
        self.registers[ptr] = answer