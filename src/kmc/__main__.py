"""Allow running KMC as a module: python -m kmc --help."""

from .cli import main

if __name__ == "__main__":
    main()
