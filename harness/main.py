"""Compatibility entry point for the Robodawn demo."""
if __package__:
    from .robodawn.main import *
else:
    from robodawn.main import *

if __name__ == "__main__":
    main()
