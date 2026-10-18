"""Native output-syntax boundary only; never reads or routes user language."""
import re


def name_phase(generated):
    # The fixed outer tool-call prefix belongs to the prompt. The first function
    # header alone can receive the learned correction, never any argument text.
    return re.fullmatch(r'\s*<function=[a-z_]*',generated) is not None
