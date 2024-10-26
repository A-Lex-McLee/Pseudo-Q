The Syntax of Sudoku Grids: Infrastructure

Sudoku is really fun -- but it's also really hard, in fact, it is NP hard.
This does not merely concern the solving of actual Sudoku puzzles, but also the grid architecture as a whole, as observed e.g. in the attempt to generate a random (valid) Grid. 
This repository is part of a side project of mine called "The syntax of Sudoko grids" (some might prefer the label 'geometry' instead of 'syntax', but I'm a linguist and I call the shots here!). It provides the basic architecture to instantiate, generate, manipulate and permute Sudoku grids. 

Notice that this is work in progress; some functionalities and parts of the documentation are still in the making; some aspects may be modified or scrapped altogether in future versions.  Feedback is always welcome & appreciated (writeTo nonintersective at gmail dot com).

I will also provide a tutorial asap, but in the meantime, in order not to completely deprive you of any sudokulogical excitement, here are some kick-off guidelines (nb: '*' --> footnote):

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
18. Do not attempt to open the .txt file manually; for inspection, try this:

>>> with open("ABC_abbreviated_0.txt", "r") as f:
>>> 
>>>         abc_collection = json.load(f)
>>> 
>>> for abc in abc_collection[:100]:
>>>     
>>>         print(abc)

19. What you see (i.e. abc_collection) is the result of .fullPermutation >>> .translate individually >>> compress tuple -> str >>> save. Via the operations mentioned above, the grids of abc_collection can be translated and permuted back and forth. NOTICE: the regular procedure produces said  3.359.232 grids compressed to strings; in memory, the corresponding .txt file takes 285.5 MB, which is too large for upload. Thus "ABC_abbreviated_0.txt" is an abbreiavted collection, containing merely 50.000 grid strings. 
20. The processing of large sets of grid collections, in turn, is the job of the class GridCollection (work in progress)
21. As mentioned, the documentation for the Grid class is not complete yet, but it should be sufficently explanatory to aid in further explorations!  




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






