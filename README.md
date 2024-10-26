The Syntax of Sudoku Grids: Infrastructure

Sudoku is really fun -- but it's also really hard, in fact, it is NP hard.
This does not merely concern the solving of actual Sudoku puzzles, but also the grid architecture as a whole, as observed e.g. in the attempt to generate a random (valid) Grid. 
This repository is part of a side project of mine called "The syntax of Sudoko grids" (even though some might prefer 'geometry instead of 'syntax'). It provides the basic architecture to instantiate, generate, manipulate and permute Sudoku grids. 

I will provide a tutorial asap, but in the meantime, in order not to completely deprive you of any sudokulogical excitement, here are some guidelines:

1. download the files (minimally, the .py files) from the repository into the same directory
2. run 'Pseudo_Q.py' in the shell/IDE of your choice
3. instantiate a Grid object: >>> grid = Grid()
4. (one way to) initialize the grid: >>> grid.insert(grd3) *1
5. show the grid coordinates: >>> grid.showFrame()  *2
6. show the grid: >>> grid.showGrid() *3
7. scan the grid box-wise: >>> grid.scanGrid()  *4


Notes:

- *1 Five valid grid tuples, called grd1, grd2, grd3, grd4, grd4, are provided to get started (free of charge!)
- *2 looks like a protocol; not easy to read
- *3 much more reader-friendly
- *4 by creating 'ad-hoc' boxes inside the grid (i.e. no overflow)
- 
- 







'Pseudo_Q' 


class Cell

class Grid

classGridCollection

class LinkedSequence (+ Node)

class KillerSolver

class OverflowAlgebra
