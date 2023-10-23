from loguru import logger

import parsing
import vms
import sys

def main():
    try:
        fname, vm_type = sys.argv[1:]
        logger.info(f"Loading {fname}")
        with open(fname) as in_f:
            program = parsing.parse_program(in_f)
        logger.info(f"Running {fname}")
        match vm_type:
            case "ref":
                vm = vms.VM(constants=program.constants)
            case "gpt":
                vm = vms.GPTVM(constants=program.constants)
            case _:
                raise NotImplementedError(vm_type)
        retval = vm.run_program(program)
        logger.info(f"Program returned {retval}")
        sys.exit(retval)
    except Exception:
        logger.exception("Error running program")

if __name__ == '__main__':
    main()