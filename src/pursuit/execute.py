import sys
import argparse
from functools import partial
import itertools
import json

from deap import gp, creator, base
from .simulator import AntSimulator, Context

parser = argparse.ArgumentParser(prog="Run")
parser.add_argument(
    "-i",
    "--input",
    dest="inputfile",
    type=argparse.FileType("r"),
    default=sys.stdin,
    required=False,
    help="filepath to a solution tree",
)
parser.add_argument(
    "-o",
    "--output",
    dest="outputfile",
    type=argparse.FileType("w"),
    default=sys.stdout,
    required=False,
    help="file which to dump the ant trace",
)
parser.add_argument(
    "--max-moves",
    dest="max_moves",
    type=int,
    default=600,
    required=False,
    help="maximum number of moves allowed",
)


def progn(*outs, ctx):
    for out in outs:
        out(ctx=ctx)


def create_primitive_set(antsim: AntSimulator) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", arity=0)
    pset.addPrimitive(
        lambda *args: partial(antsim.if_food_ahead, *args), 2, name="ifFoodAhead"
    )
    pset.addPrimitive(
        lambda *args: partial(antsim.if_food_left, *args), 2, name="ifFoodLeft"
    )
    pset.addPrimitive(
        lambda *args: partial(antsim.if_food_right, *args), 2, name="ifFoodRight"
    )
    # pset.addPrimitive(
    #     lambda *args: partial(antsim.if_food_behind, *args), 2, name="ifFoodBehind"
    # )
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addTerminal(antsim.forward, name="forward")
    pset.addTerminal(antsim.left, name="left")
    pset.addTerminal(antsim.right, name="right")
    return pset


def parse_str(text: str):
    ant_icons = {
        "^": 0 - 1j,  # up
        ">": 1 + 0j,  # right
        "v": 0 + 1j,  # down
        "<": -1 + 0j,  # left
    }
    food_icon = "#"

    assert (
        sum(map(text.count, ant_icons.keys())) == 1
    ), f"exactly one of {list(ant_icons.keys())} must be present in the environment"
    grid = [list(line) for line in map(str.strip, text.split("\n")) if line != ""]
    nrows = len(grid)
    ncols = len(grid[0]) if nrows > 0 else 0
    start_pos = (0, 0)
    start_dir = ant_icons[">"]
    for row, col in itertools.product(range(nrows), range(ncols)):
        if grid[row][col] in ant_icons.keys():
            start_pos = (row, col)
            start_dir = ant_icons[grid[row][col]]
            break
    foods = [
        (row, col)
        for row in range(nrows)
        for col in range(ncols)
        if grid[row][col] == food_icon
    ]
    return dict(
        nrows=nrows,
        ncols=ncols,
        startpos=start_pos,
        startdir=start_dir,
        foods=foods,
    )


def parse_file(filepath: str):
    with open(filepath, "r") as file:
        return parse_str(file.read())


def main() -> int:
    args = parser.parse_args()
    config = parse_file("examples/santafe.txt")
    ctx = Context(
        ncols=config["ncols"],
        nrows=config["nrows"],
        foods=config["foods"],
    )
    antsim = AntSimulator(
        ncols=config["ncols"],
        nrows=config["nrows"],
        startpos=config["startpos"],
        startdir=config["startdir"],
        foods=config["foods"],
        max_moves=args.max_moves,
    )

    pset = create_primitive_set(antsim)
    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness, steps=list)

    routine = gp.compile(args.inputfile.read(), pset=pset)
    _, _, steps = antsim.run(routine, ctx)
    json.dump(steps, fp=args.outputfile)

    return 0


if __name__ == "__main__":
    sys.exit(main())
