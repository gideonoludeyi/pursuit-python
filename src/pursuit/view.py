import argparse
import itertools
import json
from operator import iconcat
import os
import sys

parser = argparse.ArgumentParser(prog="View")
parser.add_argument(
    "-i",
    "--inputfile",
    dest="inputfile",
    type=argparse.FileType("r"),
    default=sys.stdin,
    required=False,
)
parser.add_argument(
    "-m",
    "--mapfile",
    dest="mapfile",
    type=str,
    default="examples/santafe.txt",
    required=False,
)

ant_icons = {
    "^": 0 - 1j,  # up
    ">": 1 + 0j,  # right
    "v": 0 + 1j,  # down
    "<": -1 + 0j,  # left
}
inv_ant_icons = {
    0 - 1j: "^",  # up
    1 + 0j: ">",  # right
    0 + 1j: "v",  # down
    -1 + 0j: "<",  # left
}

food_icon = "#"


def parse_str(text: str):
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


def parse_file(filepath: str | os.PathLike):
    with open(filepath, "r") as file:
        return parse_str(file.read())


def visualize(state, nrows, ncols):
    pos = state["pos"]
    icon = state["icon"]
    foods = state["foods"]
    eaten = state["eaten"]
    grid = [["." for _ in range(ncols)] for _ in range(nrows)]
    for x, y in foods:
        grid[x][y] = "#"
    for x, y in eaten:
        grid[x][y] = "?"
    grid[pos[0]][pos[1]] = icon
    return "\n".join("".join(row) for row in grid)


def vizinit(config):
    pos = config["startpos"]
    dir = config["startdir"]
    foods = config["foods"]
    grid = [["." for _ in range(config["ncols"])] for _ in range(config["nrows"])]
    for x, y in foods:
        grid[x][y] = food_icon
    grid[pos[0]][pos[1]] = inv_ant_icons[dir]
    return "\n".join("".join(row) for row in grid)


def vizfinal(steps, nrows, ncols):
    grid = [["." for _ in range(ncols)] for _ in range(nrows)]
    for step in steps:
        px, py = step["ant"]["pos"]
        dx, dy = step["ant"]["dir"]
        icon = inv_ant_icons[dx + dy * 1j]
        grid[px][py] = icon
    last_step = steps[-1]
    for food in last_step["eaten"]:
        grid[food[0]][food[1]] = "?"
    for food in last_step["foods"]:
        grid[food[0]][food[1]] = "#"
    return "\n".join("".join(row) for row in grid)


def main() -> int:
    args = parser.parse_args()
    config = parse_file(args.mapfile)
    # pos = config["startpos"]
    # dir = config["startdir"]
    # foods: list[tuple[int, int]] = config["foods"]
    # eaten = []
    # state = dict(pos=pos, icon=inv_ant_icons[dir], foods=foods, eaten=eaten)
    print(vizinit(config))
    steps: list[dict] = json.load(args.inputfile)
    print()
    print(vizfinal(steps, config["nrows"], config["ncols"]))
    # for step in steps:
    #     pos = tuple(step["ant"]["pos"])
    #     dir = step["ant"]["dir"][0] + step["ant"]["dir"][1] * 1j
    #     foods = step["foods"]
    #     eaten = step["eaten"]
    #     state = dict(pos=pos, icon=inv_ant_icons[dir], foods=foods, eaten=eaten)
    #     print("\n\n")
    #     print(visualize(state, config["nrows"], config["ncols"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
