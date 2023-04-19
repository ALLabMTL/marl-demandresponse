from dataclasses import dataclass


@dataclass
class Color:
    """
    The Color class represents a set of ANSI escape codes for coloring console output. It is a dataclass that defines class attributes for various color codes that can be used to format console output.

    Attributes:
        PURPLE: A string representing the ANSI escape code for purple color.
        CYAN: A string representing the ANSI escape code for cyan color.
        DARKCYAN: A string representing the ANSI escape code for dark cyan color.
        BLUE: A string representing the ANSI escape code for blue color.
        GREEN: A string representing the ANSI escape code for green color.
        YELLOW: A string representing the ANSI escape code for yellow color.
        RED: A string representing the ANSI escape code for red color.
        BOLD: A string representing the ANSI escape code for bold text.
        UNDERLINE: A string representing the ANSI escape code for underlined text.
        END: A string representing the ANSI escape code for resetting the text color and style.
    """

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    DARKCYAN = "\033[36m"
    BLUE = "\033[94m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"
    END = "\033[0m"
