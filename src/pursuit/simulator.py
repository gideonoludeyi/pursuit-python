import random


class Context:
    def __init__(
        self,
        nrows: int,
        ncols: int,
        foods: list[tuple[int, int]],
        *,
        seed=None,
    ) -> None:
        self.nrows = nrows
        self.ncols = ncols
        self.original_foods = list(foods)
        self.foods = list(self.original_foods)
        self.eaten_foods = []
        self.seed = seed
        self.rng = random.Random(self.seed)
        self.steps = []

    def _predator_nearby(self, antloc: tuple[int, int], foodloc: tuple[int, int]):
        manhdist = abs(antloc[0] - foodloc[0]) + abs(antloc[1] - foodloc[1])
        return manhdist <= 1

    def update(self, ant: "AntSimulator"):
        for i in range(len(self.foods)):
            if (
                self._predator_nearby(ant.pos, self.foods[i])
                and self.rng.random() <= 0.5
            ):
                dir = self.rng.choice([(1, 0), (0, 1), (-1, 0), (0, -1)])
                newloc = (
                    (self.foods[i][0] + dir[0]) % self.nrows,
                    (self.foods[i][1] + dir[1]) % self.ncols,
                )
                if newloc not in self.foods:
                    # prevent multiple preys overlapping in the same cell
                    self.foods[i] = newloc
        self.steps.append(
            dict(
                ant=dict(pos=ant.pos, dir=(ant.dir.real, ant.dir.imag)),
                foods=list(self.foods),
                eaten=list(self.eaten_foods),
            )
        )

    def eat(self, loc: tuple[int, int]):
        self.foods.remove(loc)
        self.eaten_foods.append(loc)

    def reset(self):
        self.foods = list(self.original_foods)
        self.eaten_foods = []
        self.rng = random.Random(self.seed)
        self.steps = []


class AntSimulator:
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
        steps = ctx.steps
        self._reset()
        ctx.reset()
        return n_eaten, moves, steps
