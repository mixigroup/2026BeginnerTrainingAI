# Sample Python code for beginners
# Run with: uv run src/sample.py


def greet(name: str) -> str:
    """Return a greeting message."""
    return f"Hello, {name}!"


def add(a: int, b: int) -> int:
    """Add two numbers and return the result."""
    return a + b


def fizzbuzz(n: int) -> list[str]:
    """
    Classic FizzBuzz: return a list of strings from 1 to n.
    - Multiples of 3 -> "Fizz"
    - Multiples of 5 -> "Buzz"
    - Multiples of both -> "FizzBuzz"
    - Others -> the number as a string
    """
    result = []
    for i in range(1, n + 1):
        if i % 15 == 0:
            result.append("FizzBuzz")
        elif i % 3 == 0:
            result.append("Fizz")
        elif i % 5 == 0:
            result.append("Buzz")
        else:
            result.append(str(i))
    return result


if __name__ == "__main__":
    # Greeting
    print(greet("World"))

    # Addition
    result = add(3, 5)
    print(f"3 + 5 = {result}")

    # FizzBuzz
    print("\nFizzBuzz (1 to 20):")
    for item in fizzbuzz(20):
        print(item)
