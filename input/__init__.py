# Input module: handles user parameters and CLI argument parsing.
# Think of this as the "front door" of the CAE pipeline — everything
# the user controls flows through here.
from .parameters import BeamParameters
from .cli import parse_args, validate_parameters

__all__ = ["BeamParameters", "parse_args", "validate_parameters"]
