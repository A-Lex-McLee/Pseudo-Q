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
8. generate a random grid (i):  >>> grid.generateGrid_flat()  *5, *6
9. generate a random grid (ii): >>> grid.generateGrid_deep()  *5, *6
10. rotate the grid by 90°: >>> grid.rotate()  *6
11. reflect the grid along the diagonal axis: >>> grid.diaflect() *6
12. translate the grid categorically (int <-> str): >>> grid.translate() *6
13. repeat step 12. once more *6
14. translate the grid internally (int/str <-> int/str): >>> grid.recode(encode=(9,8,7,6,5,4,3,2,1)) *6
15. translate the grid internally (int/str <-> int/str): >>> grid.recode(encode=('C','B','H','G','F','I','D','E','A',)) *6
16. go back to int-view (e.g. step 14.)
17. if you are very patient (min. 8 minutes), try the following: >>> perms = grid.fullPermutation()   ---  Congratulations! You have successfully generated the basic permutation series of a grid resulting in 3!^8*2 = 3.359.232 valid grids. *7, *8
18. Do not attempt to open the .txt files manually; for inspection, try this:
    1. | >>> with open("ABC_Perm_0.txt", "r") as f:
    2. | >>> ___   abc_collection = json.load(f)
    3. | >>> for abc in abc_collection[:100]:
    4. | >>>    print(abc)
20. 2. dfdsfds
    3. 
21. 
22. 
23.
24. s


Notes:

- 1: Five valid grid tuples, called grd1, grd2, grd3, grd4, grd4, are provided to get started (free of charge!)
- 2: looks like a protocol; not easy to read
- 3: much more reader-friendly
- 4: by creating 'ad-hoc' boxes inside the grid (i.e. no overflow)
- 5: you might wanna be patient ... 
- 6: for visualizing the result, repeat step 6. afterwards, i.e. >>> grid.showGrid()
- 7: for access, unpack the iterator (e.g. >>> perms = tuple(perms)), or iterate manually (>>> next(perms))
- 8: considering that there are 9! = 362.880 possible encoders for the .recode() method (step 14.), altogether 1.218.998.108.160 valid (surface-distinct) grids can be produced on the basis of one.
- 







'Pseudo_Q' 


class Cell

class Grid

classGridCollection

class LinkedSequence (+ Node)

class KillerSolver

class OverflowAlgebra
