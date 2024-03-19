import argparse
import itertools
import os
import random
import sys
from functools import partial

import numpy as np
from deap import algorithms, base, creator, gp, tools


class AntSimulator:
    ant_icons = {
        "^": 0 - 1j,  # up
        ">": 1 + 0j,  # right
        "v": 0 + 1j,  # down
        "<": -1 + 0j,  # left
    }
    food_icon = "#"

    def __init__(
        self,
        nrows: int,
        ncols: int,
        startpos: tuple[int, int],
        startdir: complex,
        foods: list[tuple[int, int]],
        *,
        max_moves: int,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.startpos = startpos
        self.startdir = startdir
        self.max_moves = max_moves
        self.pos = startpos
        self.dir = startdir
        self.moves = []
        self.foods = foods

    def left(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.dir *= -1j
        self.moves.append("L")
        ctx.update(self)

    def right(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.dir *= 1j
        self.moves.append("R")
        ctx.update(self)

    def forward(self, *, ctx: "Context"):
        if len(self.moves) >= self.max_moves:
            return
        self.pos = (
            (self.pos[0] + int(self.dir.imag)) % self.nrows,
            (self.pos[1] + int(self.dir.real)) % self.ncols,
        )
        if self.pos in ctx.foods:
            ctx.eat(self.pos)
        self.moves.append("F")
        ctx.update(self)

    def _has_food_ahead(self, ctx: "Context") -> bool:
        next_pos = (
            (self.pos[0] + int(self.dir.imag)) % self.nrows,
            (self.pos[1] + int(self.dir.real)) % self.ncols,
        )
        return next_pos in ctx.foods

    def if_food_ahead(self, out1, out2, *, ctx: "Context"):
        if self._has_food_ahead(ctx):
            out1(ctx=ctx)
        else:
            out2(ctx=ctx)

    def _reset(self):
        self.pos = self.startpos
        self.dir = self.startdir
        self.moves = []

    def run(self, routine, ctx: "Context"):
        ctx.reset()
        self._reset()
        while len(ctx.foods) > 0 and len(self.moves) < self.max_moves:
            routine(ctx=ctx)
        n_eaten = len(ctx.eaten_foods)
        moves = self.moves
        self._reset()
        ctx.reset()
        return n_eaten, moves

    @classmethod
    def parse_file(
        cls, filepath: str | os.PathLike, *, max_moves: int
    ) -> "AntSimulator":
        with open(filepath, "r") as file:
            return cls.parse_str(file.read(), max_moves=max_moves)

    @classmethod
    def parse_str(cls, text: str, *, max_moves: int) -> "AntSimulator":
        assert (
            sum(map(text.count, cls.ant_icons.keys())) == 1
        ), f"exactly one of {list(cls.ant_icons.keys())} must be present in the environment"
        grid = [list(line) for line in map(str.strip, text.split("\n")) if line != ""]
        nrows = len(grid)
        ncols = len(grid[0]) if nrows > 0 else 0
        start_pos = (0, 0)
        start_dir = cls.ant_icons[">"]
        for row, col in itertools.product(range(nrows), range(ncols)):
            if grid[row][col] in cls.ant_icons.keys():
                start_pos = (row, col)
                start_dir = cls.ant_icons[grid[row][col]]
                break
        foods = [
            (row, col)
            for row in range(nrows)
            for col in range(ncols)
            if grid[row][col] == cls.food_icon
        ]
        return cls(
            nrows=nrows,
            ncols=ncols,
            startpos=start_pos,
            startdir=start_dir,
            foods=foods,
            max_moves=max_moves,
        )


class Context:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        startpos: tuple[int, int],
        startdir: complex,
        foods: list[tuple[int, int]],
        *,
        max_moves: int = 600,
        rng: random.Random = random.Random(123),
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.foods = foods
        self.eaten_foods = []
        self.rng = rng

    def _predator_nearby(self, antloc: tuple[int, int], foodloc: tuple[int, int]):
        manhdist = abs(antloc[0] - foodloc[0]) + abs(antloc[1] - foodloc[1])
        return manhdist <= 3

    def update(self, ant: "AntSimulator"):
        # dirs = self.rng.choices([(1, 0), (0, 1), (-1, 0), (0, -1)], k=len(self.foods))
        for i in range(len(self.foods)):
            if (
                self._predator_nearby(ant.pos, self.foods[i])
                and self.rng.random() <= 0.6
            ):
                dir = self.rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                self.foods[i] = (
                    (self.foods[i][0] + dir[0]) % self.nrows,
                    (self.foods[i][1] + dir[1]) % self.ncols,
                )

    def eat(self, loc: tuple[int, int]):
        self.foods.remove(loc)
        self.eaten_foods.append(loc)

    def reset(self):
        self.foods = self.foods + self.eaten_foods
        self.eaten_foods = []


def progn(*outs, ctx):
    for out in outs:
        out(ctx=ctx)


def eval_artificial_ant(
    individual,
    pset: gp.PrimitiveSet,
    antsim: AntSimulator,
    ctx: Context,
    out: list[str] | None = None,
):
    routine = gp.compile(individual, pset=pset)
    n_foods = len(antsim.foods)
    n_eaten, moves = antsim.run(routine, ctx)
    if out is not None:
        out[:] = moves
    return (n_eaten / n_foods,)


def create_primitive_set(antsim: AntSimulator) -> gp.PrimitiveSet:
    pset = gp.PrimitiveSet("MAIN", arity=0)
    pset.addPrimitive(
        lambda *args: partial(antsim.if_food_ahead, *args), 2, name="ifFoodAhead"
    )
    pset.addPrimitive(lambda *args: partial(progn, *args), 2, name="prog2")
    pset.addPrimitive(lambda *args: partial(progn, *args), 3, name="prog3")
    pset.addTerminal(antsim.forward, name="forward")
    pset.addTerminal(antsim.left, name="left")
    pset.addTerminal(antsim.right, name="right")
    return pset


def eaSimpleElitism(
    population,
    toolbox,
    cxpb,
    mutpb,
    nelites,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population) - nelites)

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # include elites from previous generation
        elites = sorted(population, key=lambda ind: ind.fitness.values, reverse=True)[
            :nelites
        ]
        offspring = offspring + elites
        random.shuffle(offspring)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

    return population, logbook


parser = argparse.ArgumentParser(prog="Pursuit")
parser.add_argument(
    "-c",
    "--crossover-rate",
    dest="crossover_rate",
    type=float,
    required=False,
    default=0.9,
)
parser.add_argument(
    "-m",
    "--mutation-rate",
    dest="mutation_rate",
    type=float,
    required=False,
    default=0.1,
)
parser.add_argument(
    "-g",
    "--generatinos",
    dest="n_generations",
    type=int,
    required=False,
    default=40,
)
parser.add_argument(
    "-e", "--elites", dest="n_elites", type=int, required=False, default=0
)
parser.add_argument(
    "-p", "--popsize", dest="popsize", type=int, required=False, default=500
)
parser.add_argument(
    "--max-moves", dest="max_moves", type=int, required=False, default=600
)
parser.add_argument(
    "--tournsize", dest="tournsize", type=int, required=False, default=3
)
parser.add_argument(
    "-s", "--seed", dest="random_seed", type=int, required=False, default=123
)


def main() -> int:
    args = parser.parse_args()
    random.seed(args.random_seed)

    antsim = AntSimulator.parse_file("examples/santafe.txt", max_moves=args.max_moves)
    ctx = Context(
        ncols=antsim.ncols,
        nrows=antsim.nrows,
        startpos=antsim.startpos,
        startdir=antsim.startdir,
        foods=antsim.foods,
        max_moves=args.max_moves,
    )

    pset = create_primitive_set(antsim)

    creator.create("Fitness", base.Fitness, weights=(1.0,))
    creator.create("Individual", gp.PrimitiveTree, fitness=creator.Fitness)
    toolbox = base.Toolbox()
    toolbox.register("expr_init", gp.genFull, pset=pset, min_=3, max_=7)
    toolbox.register(
        "individual", tools.initIterate, creator.Individual, toolbox.expr_init
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", eval_artificial_ant, pset=pset, antsim=antsim, ctx=ctx)
    toolbox.register("select", tools.selTournament, tournsize=args.tournsize)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("expr_mut", gp.genHalfAndHalf, min_=3, max_=7)
    toolbox.register("mutate", gp.mutUniform, expr=toolbox.expr_mut, pset=pset)

    pop = toolbox.population(n=args.popsize)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("max", np.max)

    _, _logbook = eaSimpleElitism(
        pop,
        toolbox,
        args.crossover_rate,
        args.mutation_rate,
        args.n_elites,
        args.n_generations,
        stats,
        hof,
        verbose=True,
    )

    print(f"Best Individual: fitness={hof[0].fitness.values}, height={hof[0].height}")
    with open("best.ind", "w") as f:
        f.write(str(hof[0]))

    return 0


if __name__ == "__main__":
    sys.exit(main())
