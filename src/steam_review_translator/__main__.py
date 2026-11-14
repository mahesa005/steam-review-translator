"""Allow ``python -m steam_review_translator``."""

import sys

from steam_review_translator.cli import main

if __name__ == "__main__":
    sys.exit(main())
