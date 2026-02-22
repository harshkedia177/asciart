# Short 10-char ramp — good general purpose
STANDARD = " .:-=+*#%@"

# Minimal 4-char ramp — retro feel
MINIMAL = " .:#"

# Paul Bourke's 70-char ramp — finest granularity
DETAILED = " .'`^\",:;Il!i><~+_-?][}{1)(|\\/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

# Unicode block characters
BLOCKS = " ░▒▓█"

# Letters only — sorted by visual density
ALPHABETS = " .:ciloeCOGMWB@"

# Digits only — sorted by visual density
NUMBERS = " .17320456989"

# Mixed letters + digits — sorted by visual density
ALPHANUMERIC = " .1ico3a5mnw8MW#B@"

# Named ramp registry
RAMPS: dict[str, str] = {
    "standard": STANDARD,
    "detailed": DETAILED,
    "minimal": MINIMAL,
    "blocks": BLOCKS,
    "alphabets": ALPHABETS,
    "numbers": NUMBERS,
    "alphanumeric": ALPHANUMERIC,
}


def resolve_chars(value: str) -> str:
    """If value is a known ramp name, return the ramp string. Otherwise treat as literal."""
    return RAMPS.get(value, value)
