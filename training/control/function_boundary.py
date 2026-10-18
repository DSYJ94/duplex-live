"""Native output-syntax boundary only; never reads or routes user language."""
import re


def name_phase(generated):
    # The fixed outer tool-call prefix belongs to the prompt. The first function
    # header alone can receive the learned correction, never any argument text.
    return re.fullmatch(r'\s*<function=[a-z_]*',generated) is not None


def target_positions(tokens,decode):
    """Predict only where the deployed hook would apply its correction.

    Position i predicts tokens[i] using already-arrived tokens[:i]. The first
    output uses the untouched prefill projection; arguments/EOS remain masked.
    Decoding the actual token prefixes also handles tokens spanning delimiters.
    """
    positions=[i for i in range(1,len(tokens)) if name_phase(decode(tokens[:i]))]
    if not positions:raise ValueError('No actual function-name correction positions')
    return positions
