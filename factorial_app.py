"""A tiny CLI app that calculates the factorial of a non-negative integer."""

from __future__ import annotations

import argparse


def factorial(value: int) -> int:
    if value < 0:
        raise ValueError("factorial is undefined for negative numbers")
    result = 1
    for current in range(2, value + 1):
        result *= current
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Calculate the factorial of a non-negative integer.")
    parser.add_argument("value", type=int, help="A non-negative integer to factorize")
    args = parser.parse_args()

    try:
        print(factorial(args.value))
        return 0
    except ValueError as error:
        print(f"Error: {error}", file=None)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
