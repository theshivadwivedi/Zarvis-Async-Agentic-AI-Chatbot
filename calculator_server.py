
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Calculator")


def _validate_numbers(*values) -> str | None:
    """Return an error message if any value is not a finite real number, else None."""
    import math
    for v in values:
        if not isinstance(v, (int, float)):
            return f"ERROR: Expected a number but got '{v}' ({type(v).__name__})."
        if math.isnan(v):
            return "ERROR: NaN is not a valid input."
        if math.isinf(v):
            return "ERROR: Infinity is not a valid input."
    return None


@mcp.tool()
def add(a: float, b: float) -> str:
    """
    Add two numbers and return the result.
    Returns an error string if inputs are invalid.
    """
    err = _validate_numbers(a, b)
    if err:
        return err
    result = a + b
    return f"{a} + {b} = {result}"


@mcp.tool()
def subtract(a: float, b: float) -> str:
    """
    Subtract b from a and return the result.
    Returns an error string if inputs are invalid.
    """
    err = _validate_numbers(a, b)
    if err:
        return err
    result = a - b
    return f"{a} - {b} = {result}"


@mcp.tool()
def multiply(a: float, b: float) -> str:
    """
    Multiply two numbers and return the result.
    Returns an error string if inputs are invalid.
    """
    err = _validate_numbers(a, b)
    if err:
        return err
    result = a * b
    return f"{a} × {b} = {result}"


@mcp.tool()
def divide(a: float, b: float) -> str:
    """
    Divide a by b and return the result.
    Returns an error string if b is zero or inputs are invalid.
    """
    err = _validate_numbers(a, b)
    if err:
        return err
    if b == 0:
        return "ERROR: Division by zero is undefined."
    result = a / b
    return f"{a} ÷ {b} = {result}"


@mcp.tool()
def percentage(value: float, total: float) -> str:
    """
    Calculate what percentage 'value' is of 'total'.
    Example: percentage(25, 200) → '25 is 12.5% of 200'
    Returns an error string if total is zero or inputs are invalid.
    """
    err = _validate_numbers(value, total)
    if err:
        return err
    if total == 0:
        return "ERROR: Total cannot be zero when calculating a percentage."
    result = round((value / total) * 100, 6)
    return f"{value} is {result}% of {total}"


if __name__ == "__main__":
    mcp.run(transport="stdio")