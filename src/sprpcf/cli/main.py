"""Main entry point for the sprpcf CLI tool."""

import sys
import argparse
from ..version import __version__
from .commands import cmd_init, cmd_validate, cmd_geometry, cmd_mesh, cmd_run


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="sprpcf",
        description="SPR-PCF Studio: Universal Computational Framework for SPR Biosensor Simulation",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", help="Available subcommands")

    # init
    p_init = subparsers.add_parser("init", help="Create a sample design.yaml file")
    p_init.add_argument("-o", "--output", help="Output file path (default: design.yaml)")
    p_init.set_defaults(func=cmd_init)

    # validate
    p_val = subparsers.add_parser("validate", help="Validate a YAML/JSON configuration file")
    p_val.add_argument("config", help="Path to config file")
    p_val.set_defaults(func=cmd_validate)

    # geometry
    p_geom = subparsers.add_parser("geometry", help="Render geometry cross-section")
    p_geom.add_argument("config", help="Path to config file")
    p_geom.add_argument("-o", "--output", help="Output image file path")
    p_geom.set_defaults(func=cmd_geometry)

    # mesh
    p_mesh = subparsers.add_parser("mesh", help="Generate and render mesh")
    p_mesh.add_argument("config", help="Path to config file")
    p_mesh.add_argument("-o", "--output", help="Output image file path")
    p_mesh.set_defaults(func=cmd_mesh)

    # run
    p_run = subparsers.add_parser("run", help="Run simulation study")
    p_run.add_argument("config", help="Path to config file")
    p_run.add_argument("-o", "--output-dir", help="Results directory (default: results)")
    p_run.set_defaults(func=cmd_run)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()

