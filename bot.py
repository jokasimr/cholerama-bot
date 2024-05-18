# SPDX-License-Identifier: BSD-3-Clause
import os
import random

from typing import Optional, Tuple

import numpy as np

from cholerama import Positions, helpers

AUTHOR = "Frothing puffer"  # This is your team name
SEED = None  # Set this to a value to make runs reproducible


def concatenate(*ps):
    return Positions(
        x=np.concatenate([p.x for p in ps]),
        y=np.concatenate([p.y for p in ps])
    )


def rotate(p):
    return mirror(Positions(x=p.y, y=p.x), 'x')


def random_orientation(p):
    for i in range(random.randint(0, 3)):
        p = rotate(p)
    if random.randint(0, 1):
        p = mirror(p, 'x')
    if random.randint(0, 1):
        p = mirror(p, 'y')
    return p


def mirror(p, dim):
    m = max(np.max(p.x), np.max(p.y))
    if dim == 'x':
        return Positions(x=m - p.x, y=p.y)
    if dim == 'y':
        return Positions(y=m - p.y, x=p.x)
    assert False


def shift(s, p):
    return Positions(s[0] + p.x, s[1] + p.y)


def local_path(fname):
    return os.path.join(os.path.dirname(__file__), fname)


def plaintext_to_pattern(fname):
    with open(local_path(fname)) as f:
        lines = f.readlines()
    lines = [line for line in lines if not line.startswith('!')]
    nonzero = sum(c == 'O' for line in lines for c in line)
    x = np.empty(nonzero, dtype='int')
    y = np.empty(nonzero, dtype='int')
    k = 0
    for i, line in enumerate(lines):
        for j, c in enumerate(line):
            if c == 'O':
                y[k] = len(lines) - i - 1
                x[k] = j
                k += 1
    return Positions(x=x, y=y)


def rle_to_pattern(fname):
    with open(local_path(fname)) as f:
        lines = f.readlines()
    lines = [line for line in lines if not line.startswith('#')]

    xs = []
    ys = []
    num = ''
    x = 0
    y = 0
    for line in lines[1:]:
        for c in line:
            if c in '0123456789':
                num += c
            if c == 'b':
                x += 1 if num == '' else int(num)
                num = ''
            elif c == 'o':
                for i in range(1 if num == '' else int(num)):
                    xs.append(x)
                    ys.append(y)
                    x += 1
                num = ''
            elif c == '$':
                y += 1 if num == '' else int(num)
                num = ''
                x = 0
            elif c == '!':
                return Positions(
                    x=np.array(xs, dtype='int'),
                    y=np.array(ys, dtype='int')
                )
    assert False


class Bot:
    """
    This is the bot that will be instantiated for the competition.

    The pattern can be either a numpy array or a path to an image (white means 0,
    black means 1).
    """

    def __init__(
        self,
        number: int,
        name: str,
        patch_location: Tuple[int, int],
        patch_size: Tuple[int, int],
    ):
        """
        Parameters:
        ----------
        number: int
            The player number. Numbers on the board equal to this value mark your cells.
        name: str
            The player's name
        patch_location: tuple
            The i, j row and column indices of the patch in the grid
        patch_size: tuple
            The size of the patch
        """
        self.number = number  # Mandatory: this is your number on the board
        self.name = name  # Mandatory: player name
        self.color = None  # Optional
        self.patch_location = patch_location
        self.patch_size = patch_size

        self.rng = np.random.default_rng(SEED)

        '''
        self.pattern = concatenate(
            mirror(rle_to_pattern('gosperglidergun.rle'), 'y'),
            shift(
                (self.patch_size[0] - 40, self.patch_size[1] - 40),
                mirror(rotate(rotate(rle_to_pattern('gosperglidergun.rle'))), 'y'),
            ),
        )
        self.pattern = concatenate(
                mirror(rle_to_pattern('blinkerpuffer1.rle'), 'y'),
            shift(
                (self.patch_size[0] - 40, self.patch_size[1] - 40),
                mirror(rotate(rotate(rle_to_pattern('blinkerpuffer1.rle'))), 'y'),
            ),
        )
        self.pattern = self.center(rle_to_pattern('backrake1nohwsspuffer.rle'))
        self.pattern = concatenate(
                mirror(rle_to_pattern('pufferfish.rle'), 'y'),
            shift(
                (self.patch_size[0] - 40, self.patch_size[1] - 40),
                mirror(rotate(rotate(rle_to_pattern('pufferfish.rle'))), 'y'),
            ),
        )
        self.pattern = concatenate(
            mirror(rle_to_pattern('blocklayingswitchenginepredecessor.rle'), 'y'),
            shift(
                (self.patch_size[0] - 40, self.patch_size[1] - 40),
                mirror(rotate(rotate(rle_to_pattern('blocklayingswitchenginepredecessor.rle'))), 'y'),
            ),
        )
        '''
        self.pattern = concatenate(
            mirror(rle_to_pattern('pufferfish.rle'), 'y'),
            shift(
                (self.patch_size[0] - 40, self.patch_size[1] - 40),
                mirror(rotate(rotate(rle_to_pattern('pufferfish.rle'))), 'y'),
            ),
            self.center(rle_to_pattern('10cellinfinitegrowth.rle'))
        )


        

    def center(self, pattern):
        return shift(
            (self.patch_size[1] // 2, self.patch_size[0] // 2),
            pattern
        )

    def iterate(
        self, iteration: int, board: np.ndarray, patch: np.ndarray, tokens: int
    ) -> Optional[Positions]:
        """
        This method will be called by the game engine on each iteration.

        Parameters:
        ----------
        iteration : int
            The current iteration number.
        board : numpy array
            The current state of the entire board.
        patch : numpy array
            The current state of the player's own patch on the board.
        tokens : list
            The list of tokens on the board.

        Returns:
        -------
        An object containing the x and y coordinates of the new cells.
        """
        if iteration < 500:
            return

        #pattern = random_orientation(rle_to_pattern('gosperglidergun.rle'))
        pattern = random_orientation(rle_to_pattern('10cellinfinitegrowth.rle'))

        if tokens >= len(pattern):
            # Pick a random empty region of size 3x3 inside my patch
            empty_regions = helpers.find_empty_regions(patch, (36, 36))
            nregions = len(empty_regions)
            if nregions == 0:
                return None

            region = empty_regions[self.rng.integers(0, nregions)]
            return shift(
                (region[1], region[0]),
                pattern,
            )
