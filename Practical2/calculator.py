import math
import re


def process_query(query):
    query = query.lower()

    # Add
    match = re.search(r"add\s+(\d+)\s+and\s+(\d+)", query)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        return f"Answer: {a + b}"

    # Multiply
    match = re.search(r"multiply\s+(\d+)\s+and\s+(\d+)", query)
    if match:
        a = int(match.group(1))
        b = int(match.group(2))
        return f"Answer: {a * b}"

    # Square Root
    match = re.search(r"(?:square root of|find square root of)\s+(\d+)", query)
    if match:
        number = int(match.group(1))
        return f"Answer: {math.sqrt(number)}"

    return "Sorry, I can only perform addition, multiplication, and square root."