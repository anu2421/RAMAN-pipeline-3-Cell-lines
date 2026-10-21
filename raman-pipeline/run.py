import argparse
import importlib.util
import os

from ramanpipe import config


def load_settings(path):
    spec = importlib.util.spec_from_file_location("experiment", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SETTINGS


def main():
    parser = argparse.ArgumentParser(description="Single-cell Raman analysis: Control vs E. coli vs S. aureus")
    parser.add_argument("experiment", help="settings file, e.g. experiments/heat_inactivated.py")
    parser.add_argument("--output", help="override the output folder")
    parser.add_argument("--permutations", type=int, help="override the number of permutations")
    args = parser.parse_args()

    settings = load_settings(args.experiment)
    if args.output:
        settings["OUTPUT_DIR"] = os.path.abspath(args.output)
    if args.permutations:
        settings["N_PERMUTATIONS"] = args.permutations
    config.configure(**settings)

    from ramanpipe.pipeline import main as run
    run()


if __name__ == "__main__":
    main()
