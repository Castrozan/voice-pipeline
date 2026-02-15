import logging

RESET = "\033[0m"
DIM = "\033[2m"
BOLD = "\033[1m"
RED = "\033[31m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
CYAN = "\033[36m"
MAGENTA = "\033[35m"

LEVEL_COLORS = {
    logging.DEBUG: DIM,
    logging.INFO: GREEN,
    logging.WARNING: YELLOW,
    logging.ERROR: RED,
    logging.CRITICAL: RED + BOLD,
}


class ColoredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        color = LEVEL_COLORS.get(record.levelno, "")
        level = record.levelname
        time = self.formatTime(record, self.datefmt)
        name = record.name.split(".")[-1]
        msg = record.getMessage()

        if "State:" in msg and "->" in msg:
            msg = f"{BOLD}{CYAN}{msg}{RESET}"
        elif "Wake word detected" in msg:
            msg = f"{BOLD}{MAGENTA}{msg}{RESET}"
        elif "Transcript:" in msg and "interim" not in msg:
            msg = f"{CYAN}{msg}{RESET}"
        elif "Utterance:" in msg:
            msg = f"{BOLD}{GREEN}{msg}{RESET}"
        elif "Speech detected" in msg:
            msg = f"{MAGENTA}{msg}{RESET}"
        elif "Barge-in" in msg:
            msg = f"{BOLD}{YELLOW}{msg}{RESET}"
        elif record.levelno == logging.DEBUG:
            msg = f"{DIM}{msg}{RESET}"
        elif record.levelno >= logging.WARNING:
            msg = f"{color}{msg}{RESET}"

        return f"{DIM}{time}{RESET} {color}{level:<5}{RESET} {DIM}{name:<16}{RESET} {msg}"
