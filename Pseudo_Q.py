#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
SUDOKU GRIDs 

The Syntax of Sudoku Grids, Part A: Infrastructure 
 
Created in Spring 2020, revised

@author: alexanderpfaff

"""

from __future__ import annotations
from itertools import permutations, combinations, product
from typing import Optional, TypeVar,List, Tuple, Set, Dict, Collection
from random import randint, shuffle #, choice
from math import factorial, sqrt
from copy import deepcopy
from dataclasses import dataclass
from collections.abc import Callable
import FunX  as fx
import numpy as np



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

V = TypeVar('V', str, int) 

@dataclass
class Cell: 
    """
    Minimal unit of a grid structure; contains the actual value and 
    the grid coordinates: 
        run -- running number 
        row -- row number 
        col -- column number 
        box -- box number 
        box_row -- wrapper structure comprising rows of box size 
        box_col -- wrapper structure comprising columns of box size 
        
    """
    run: int 
    row: int
    col: int 
    box_row: int 
    box_col: int 
    box: int 
    val: V = 0
    
    
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

# Functional parameters <==> functions to be passed in as arguments to certain
#                             methods of the Grid class
#                             (e.g. gridOut, )
#                        ==> they return the value of the eponymous attribute
#                             of some cell

def row(c: Cell) -> int:
    return c.row

def col(c: Cell) -> int:
    return c.col 

def box(c: Cell) -> int:
    return c.box 

def box_row(c: Cell) -> int:
    return c.box_row 

def box_col(c: Cell) -> int:
    return c.box_col 


# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * 

class Grid:
    """ 
        Provides a (baseNumber x baseNumber) X (baseNumber x baseNumber) Sudoku Grid,
        default setting: baseNumber = 3 ==> classical 9 X 9 Sdoku grid (recommended!);
        The constructor generates a base grid (baseGrid), which is effectively
        a coordinate system of Cell objects with the coordinates:
           -- run: running number (1 to baseNumber**4) from top-left to bottom-right,
           -- col: column number (1 to baseNumber**2) from left to right,
           -- row: row number (1 to baseNumber**2) top-down,
           -- box: box number (1 to baseNumber**2) from top-left to bottom-right,
           -- val: the actual value in a valid Sudoku grid; default setting: 0
 
           -- box_col: number of column with box-width (1 to baseNumber), left to right,
           -- box_row: number of row with box-width (1 to baseNumber), top-down.
           
           An illustration; the following grid: 
               
                *************************************
                * 2 | 6 | 4 * 3 | 8 | 9 * 5 | 1 | 7 *
                *-----------*-----------*-----------*
                * 5 | 1 | 7 * 6 | 4 | 2 * 9 | 8 | 3 *
                *-----------*-----------*-----------*
                * 3 | 8 | 9 * 7 | 5 | 1 * 4 | 6 | 2 *
                *************************************
                * 4 | 2 | 6 * 5 | 1 | 7 * 3 | 9 | 8 *
                *-----------*-----------*-----------*
                * 9 | 3 | 8 * 2 | 6 | 4 * 1 | 7 | 5 *
                *-----------*-----------*-----------*
                * 1 | 7 | 5 * 8 | 9 | 3 * 6 | 2 | 4 *
                *************************************
                * 6 | 4 | 2 * 1 | 3 | 8 * 7 | 5 | 9 *
                *-----------*-----------*-----------*
                * 7 | 5 | 3 * 9 | 2 | 6 * 8 | 4 | 1 *
                *-----------*-----------*-----------*
                * 8 | 9 | 1 * 4 | 7 | 5 * 2 | 3 | 6 *
                *************************************
                
               has a.o. the following coordinates (matrix notation): 
                   
                row 1:      [2, 6, 4, 3, 8, 9, 5, 1, 7]  
                col 2:      [6, 1, 8, 2, 3, 7, 4, 5, 9]T
                box 3:      [5, 1, 7,
                             9, 8, 3,
                             4, 6, 2]
                box_row 3: [[6, 4, 2, 1, 3, 8, 7, 5, 9]
                            [7, 5, 3, 9, 2, 6, 8, 4, 1]
                            [8, 9, 1, 4, 7, 5, 2, 3, 6]]
                col_row 2: [[3, 6, 7, 5, 2, 8, 1, 9, 4]T
                            [8, 4, 5, 1, 6, 9, 3, 2, 7]T
                            [9, 2, 1, 7, 4, 3, 8, 6, 5]T]
                    
                         
           The constructor initializes a grid with coordinates calculated from {baseNumer}
           with values set to zero. 
           The class provides a method to initialize the values via insertion 
           where the inserted object must be a sequential container object (tuple, list, string)
           containing legal value types, viz. int or str. 
           
           Further functionalities are included, such as methods for
             -- visualizing (= printing out) the grid,
             -- generating random grids, 
             -- geometric manipulation of the grid,
             -- permuting the grid coordinates,
                  notably, for generating the full permutation series of a given grid 
                  resulting in a collection of {baseNumber}!^8 * 2 grid permutations
                  (for baseNumber=3 <=> classical Sodoku grid, this means 3.359.232 grids!)
             -- alphabetizing the grid <==> de-/encoding the grid alphabetically.              
    """
    def __init__(self, baseNumber: int = 3):
        self.BASE_NUMBER: int = baseNumber
        self.DIMENSION: int = baseNumber**2
        self.SIZE: int = baseNumber**4
        self.zeroSeq: Tuple[0] = tuple(0 
                                       for i in range(self.SIZE))        
        """ Algorithm to calculate the grid coordinates. """
        self.baseGrid: Tuple[Cell] = tuple(
            Cell(run=(self.DIMENSION)*out + self.BASE_NUMBER*mid + inn + 1,
                  row=out + 1,
                  col=self.BASE_NUMBER*mid + inn + 1,
                  box_row=out//self.BASE_NUMBER + 1,
                  box_col=mid + 1,
                  box=(out//self.BASE_NUMBER)*self.BASE_NUMBER + mid + 1) 
             for out in range(self.DIMENSION)
             for mid in range(self.BASE_NUMBER)
             for inn in range(self.BASE_NUMBER) 
            )
        
    def __str__(self) -> str:
        return f'Sudoku_Grid:  ({self.BASE_NUMBER} x {self.BASE_NUMBER})  X  ({self.BASE_NUMBER} x {self.BASE_NUMBER})'

    def __repr__(self) -> str:
        return "<class 'Grid'>"

     
# * * * * * * * * * * * * * *  INVENTORY  * * * * * * * * * * * * * * * * * * * 

    """ 0.    INSERT a GRID """

    def insert(self, seq: Tuple[V]) -> None:
        """
        inserts a sequence of characters (numbers, letters ...) and 
        translates it into a grid structure.
        
        @requires seq is an ordered iterable (list, tuple, str)!  
        @requires len(seq) == self.BASE_NUMBER**4
        """
        if len(seq) != len(self.baseGrid):
            raise ValueError(f"The sequence submitted does not contain the required number of cells: {len(self.baseGrid)}")
        for i in range(len(self.baseGrid)):
            self.baseGrid[i].val = seq[i]

    def setZero(self) -> None:
        """
        Empties the grid by setting all values to zero
        """
        self.insert(self.zeroSeq)


    """ A.    VISUAL """

    def showFrame(self) -> None:
        """prints the current grid (= baseGrid) as a sequence of coordinates """
        for cell in self.baseGrid:
            outStr: str = f'run: {cell.run:2};   '  \
                + f'row: {cell.row};   '            \
                + f'col: {cell.col};   '            \
                + f'box: {cell.box};   '            \
                + f'box_col: {cell.box_col};   '    \
                + f'box_row: {cell.box_row};   '    \
                + f'value: {cell.val} '    
            print(outStr)

    def showGrid(self) -> None:      
        """prints out the current grid as Sudoku grid """
        grdLen: int = (self.DIMENSION * 2 + self.BASE_NUMBER*2 + 1)
        print()
        print("-" * grdLen)
        for i in range(self.DIMENSION):
            print("|", end=" ")
            for u in range(self.DIMENSION):
                print(self.baseGrid[self.DIMENSION*i+u].val, end=" ")
                if (self.DIMENSION*i+u) % self.BASE_NUMBER == 2:
                    print("|", end=" ")
            print()
            if (i+1) % self.BASE_NUMBER == 0:
                print("-" * grdLen)


    """ B.    VALUES / Value TUPLES by COORDINATE """
    
    def getRun(self, r: int, c: int) -> int:
        """ calculates the running number from row number r and column number c """
        return (r-1) * self.DIMENSION + c
        
    def getCell_fromRun(self, r: int) -> Cell:
        """ returns the cell with running number r """ 
        if r < 1 or r > self.BASE_NUMBER**4:
            raise ValueError(f"Invalid running number (choose 1 - {self.BASE_NUMBER**4})")
        return deepcopy(self.baseGrid[r-1])
        
    
    def getRelativeBox(self, topLeft: int) -> np.ndarray:
        """ returns a BASE_NUMBER X BASE_NUMBER value box 
            starting from running number {topLeft} 
            (-> top-left corner of the prospective box) 
        """ 
        overFlow: fx.OverflowAlgebra = fx.OverflowAlgebra(self.DIMENSION)
        out: List[V] = []
        for relativeRow in range(self.BASE_NUMBER):
            _row: int = (topLeft - 1) // self.DIMENSION + 1 + relativeRow
            for relativeCol in range(self.BASE_NUMBER):
                _col: int = overFlow.modPlus(topLeft + relativeCol) 
                _run: int = self.getRun(_row, _col)
                
                cell: Cell = self.getCell_fromRun(_run)
                out.append(cell.val)
                    
        boxAsArray: np.ndarray = np.asarray(out)
        return boxAsArray.reshape([self.BASE_NUMBER, self.BASE_NUMBER]) 
    
    

    def gridRow(self, r: int) -> Tuple[int]:
        """ returns the values in row r as tuple """ 
        if r < 1 or r > self.DIMENSION:
            raise ValueError(f"Invalid row number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.row == r)

    def gridCol(self, c: int) -> Tuple[int]:
        """ returns the values in column c as tuple """ 
        if c < 1 or c > self.DIMENSION:
            raise ValueError(f"Invalid column number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.col == c)

    def gridBox(self, b: int) -> Tuple[int]:
        """ returns the values in box b as tuple """ 
        if b < 1 or b > self.DIMENSION:
            raise ValueError(f"Invalid box number (choose 1 - {self.DIMENSION})")
        return tuple(cell.val
                for cell in self.baseGrid
                if cell.box == b)
        
    def gridOut(self, coord: Callable[Cell, int] = row) -> Tuple[int]:
        """ 
        returns the <entire> current grid as a tuple; 
        legal arguments: 
            -- row (default): sorted row-wise (= by running number)
                                      ==> de facto IS the grid,
            -- col:           sorted column-wise (= top-down),
            -- box:           sorted box-wise (box-internally: row-wise)
        
        >> NB: if row is a valid grid,
        >>      -->  col produces a valid grid 
        >>      -->  box does not (normally!?) produce a valid grid
        """
        out: list[int] = []
        for i in range(1, self.DIMENSION + 1):
            for cell in self.baseGrid:
                if coord(cell) == i:
                    out.append(cell.val)   
        return tuple(out)




    """ C.    RANDOM GRID GENERATION 
          ==> with subsequent insertion 
                (i.e. the current grid will be overwritten; 
                 for grid generation without insertion, 
                 see external functions  @generateRndGrid_cell,
                                         @generateRndGrid_deep  )
    """
    
    def generateGrid_flat(self) -> None: 
        """ ==> calls external function  @generateRndGrid_cell
 
            Generates a grid assignment with a valid digit distribution at random,
            and inserts it into the current Grid object.
            Sloppy version, does not use backtracking  
            --> can result in a quick output (or not), but it is possible
            that it does not produce any output at all because the procedure 
            >> runs out of options << as it were. 
            (for posssible fix, @see generateRndGrid_cell)        
        """
        sequence: Tuple[int] = generateRndGrid_cell(self.BASE_NUMBER)
        self.insert(sequence)

    def generateGrid_deep(self) -> None: 
        """ ==> calls external function  @generateRndGrid_deep
        
            Generates a grid assignment with a valid digit distribution at random,
            and inserts it into the current Grid object.
            Systematic semi-recursive version that implements backtracking 
            --> will always return a valid solution;
            but may take a bit more time to do so. 
        """
        sequence: Tuple[int] = generateRndGrid_deep(self.BASE_NUMBER)
        self.insert(sequence)




    """ D.    CHECK the GRID   """

    def gridCheckZero(self):
        """
        checks whether the current grid contains no more than one occurrence of every value 
        in every row, every column and every box, respectively. 
        Notably, it returns true even if the respective value occurs zero times 
        in any row, column or box. 
        """
        values: list[int] = list(range(1, self.DIMENSION+1))
        for val in values:
            for coord in range(1,self.DIMENSION+1):
                if self.gridBox(coord).count(val) > 1:
                    return False
                if self.gridCol(coord).count(val) > 1:
                    return False
                if self.gridRow(coord).count(val) > 1:
                    return False
        return True           
            
    def gridCheck(self):
        """
        checks whether the current grid is valid;
        in a a valid grid, every row, every colum and every box
        contains every character exactly once each 
        """
        values: Set[V] = set(self.gridOut())
        assert len(values) == self.DIMENSION
        for val in values:
            for coord in range(1,self.DIMENSION+1):
                if self.gridBox(coord).count(val) != 1:
                    return False
                if self.gridCol(coord).count(val) != 1:
                    return False
                if self.gridRow(coord).count(val) != 1:
                    return False
        return True           



    """ E.    SYMMETRY / GEOMETRY"""

    def rotate(self) -> None:
        """ 
            rotate the current grid 90°clockwise
            Procedure: insert reversed column values into the corresponding row.
        """
        rotation: Tuple[int] = [reverseCol_toRow
                                for coordRange in range(1,self.DIMENSION + 1)
                                for reverseCol_toRow in [self.gridCol(coordRange)[-reverseIdx]
                                                         for reverseIdx in range(1,self.DIMENSION + 1)]]
        self.insert(rotation)


    def diaflect(self) -> None:
        """ 
            reflects the grid diagonally along top-left <=> bottom-right axis;
            (= transpose in matrix terms).
        """
        self.insert(self.gridOut(col))
        

    def seqOverflow(self, coord: int = row) -> Tuple[int]:
        """
            Sequential overflow;
            moves running number += 1, with final+1 to initial position
        """
        movedSeq: list[int] = list(self.gridOut(coord))
        movedSeq.insert(0, movedSeq.pop(self.BASE_NUMBER**4-1))
        return tuple(movedSeq)
    
    
    def scanGrid(self) -> Tuple[Tuple[int, np.ndarray, int]]:
        """
        Scans the current grid box-wise (BASE_NUMBER X BASE_NUMBER) without overflow; 
        compares the sum of the current box to the currently legal sum of boxes ('kleiner Gauss')

        Returns
        -------
        scannedGrid : TYPE
        """
        legalBoxSum: int = (self.DIMENSION * (self.DIMENSION +1)) // 2
        scannedGrid: List[Tuple[int, np.ndarray, int]] = []
        for _row in range(1, (self.DIMENSION - (self.BASE_NUMBER-1)) + 1):
            for _col in range(1, (self.DIMENSION - (self.BASE_NUMBER-1)) + 1):
                _run = self.getRun(_row, _col)
                _box = self.getRelativeBox(_run) 
                currentBoxSum: int = sum(sum(_box))                
                out: Tuple[int, np.ndarray, int] = (_run, _box, legalBoxSum-currentBoxSum)
                scannedGrid.append(out)
        return tuple(scannedGrid)



    def scanGridOverflow(self) -> Tuple[Tuple[int, np.ndarray, int]]:
        """
        Scans the current grid box-wise (BASE_NUMBER X BASE_NUMBER) along the row-axis
        WITH row overflow; 
        compares the sum of the current box to the currently legal sum of boxes ('kleiner Gauss')

        Returns
        -------
        scannedGrid :    Tuple[Tuple[int, np.ndarray, int]]
            DESCRIPTION:  Tuple<running number, relative Bos, sum difference>

        """
        legalBoxSum: int = (self.DIMENSION * (self.DIMENSION +1)) // 2
        scannedGrid: List[Tuple[int, np.ndarray, int]] = []
        
        for _run in range(1, (self.DIMENSION**2 - (self.BASE_NUMBER-1)* self.DIMENSION) + 1):
            _box = self.getRelativeBox(_run) 
            currentBoxSum: int = sum(sum(_box))                
            out: Tuple[int, np.ndarray, int] = (_run, _box, legalBoxSum-currentBoxSum)
            scannedGrid.append(out)            
        return tuple(scannedGrid)






    """ F.    PERMUTE (parts of) the GRID """

    def _permuteCoord(self, pos: int = 1) -> Tuple[Tuple[int]]:
        """
        _aux_method@permutePos
        
        produces permutations of coordinates within the given positional 
        dimension; possible values for pos: 1 - self.BASE_NUMBER
        """
        coordinateSpan: List[int] = list(range(1,self.BASE_NUMBER + 1))
        lowestIndex: int = (pos-1) * self.BASE_NUMBER
        coordinateRange: list[int] = [lowestIndex + i
                           for i in coordinateSpan]
        return tuple(permutations(coordinateRange))

    def _permuteDiff(self, pos: int = 1):
        """
        _aux_method@permutePos
        
        calculates the differences between old and new positions, and produces
        permutations of the respective differences
        """
        out: List[Tuple[int]] = []
        per_mutations: Tuple[Tuple[int]] = self._permuteCoord(pos=pos)
        base: Tuple[int] = per_mutations[0]
        for perm in per_mutations:
            out.append(tuple(perm[i] - base[i]
                             for i in range(len(perm))))
        return out

    def permutePos(self, pos: int = 1, grdCoord: Callable[Cell, int] = col) -> Tuple[Tuple[int]]:
        """
        creates the self.BASE_NUMBER! permutations of grdCoord in the curent grid
        within the positional dimension pos


        Parameters
        ----------
        pos : int, optional
                position of the higher unit of the {grdCoord} coordinate, i.e.
                    * for row & col ==> box_row & box_col, respectively;
                        thus legal values are 1, 2 .. BASE_NUMBER        
                    * for box_row & box_col ==> grid.DIMENSION;
                        thus the only legal value is 1        
        grdCoord : Callable[Cell, int], optional
                    Functional parameter to specify the coordinate to be permuted:
                    row, col (= default), box_row, box_col. 

        Returns :  
        -------
        TYPE:  Tuple[Tuple[int]]
            contains the grid permutations for {grd} as tuples.
        """
        out = []
        per_mutations = self._permuteCoord(pos=pos)
        permute_Diffs = self._permuteDiff(pos=pos)
        base = per_mutations[0]
        for perm in permute_Diffs:
            tempGrid = list(self.gridOut()) 
            for cells in self.baseGrid:
                for i in range(self.BASE_NUMBER):
                    if grdCoord(cells) == base[i]:
                        if grdCoord == col:
                            diff = perm[i] * self.BASE_NUMBER**0
                        if grdCoord == box_col:
                            diff = perm[i] * self.BASE_NUMBER**1
                        if grdCoord == row:
                            diff = perm[i] * self.BASE_NUMBER**2
                        if grdCoord == box_row:
                            diff = perm[i] * self.BASE_NUMBER**3
                        ix = cells.run + diff
                        tempGrid[ix-1] = cells.val
            out.append(tuple(tempGrid))
        return tuple(out)


    # method@permuteCols & method@permuteRows are the two core ingredients of the 
    # (full) grid permutation procedure; 
    # each produces the 3!^4 = 1296 possible permutations of the respective dimension;
    # combined they result in 1296 x 1296 = 1.679.616 grid permuations (@permuteGrid)
    # currently, they are tailor-made for classical Sudoku (9 X 9) grids 
    # and do not work for arbitrary BASE_NUMBERs != 3     
    def permuteCols(self):
        saveGrid = self.gridOut()    
        out = []
        for box_grid in self.permutePos(1, grdCoord=box_col):
            self.insert(box_grid)
            for col1 in self.permutePos(1, grdCoord=col):
                self.insert(col1)
                for col2 in self.permutePos(2, grdCoord=col):
                    self.insert(col2)
                    for col3 in self.permutePos(3, grdCoord=col):
                        out.append(col3)
        self.insert(saveGrid)
        return out

    def permuteRows(self):
        saveGrid: Tuple[V] = self.gridOut()    
        out: List[V] = []
        for row_grid in self.permutePos(1, grdCoord=box_row):
            self.insert(row_grid)
            for row1 in self.permutePos(1, grdCoord=row):
                self.insert(row1)
                for row2 in self.permutePos(2, grdCoord=row):
                    self.insert(row2)
                    for row3 in self.permutePos(3, grdCoord=row):
                        out.append(row3)
        self.insert(saveGrid)
        return out

    def permuteGrid(self) -> iter[Tuple[V]]:
        """ performs a full permutation of a grid;
            for a classical 9 X 9 Sudoku grid, this means: 
            -- 3 X 3 rows + 3 box_rows: (3!)**4 = 1296 
            -- 3 X 3 cols + 3 box_cols: (3!)**4 = 1296
            ==> (3!)**8 = 1296 x 1296 = 1.679.616 grids

            NB: this will take a while ... (ca. 5-6 min)!!!
        """
        saveGrid: Tuple[V] = self.gridOut()    
        out: List[V] = []
        for cols in self.permuteCols():
            self.insert(cols)
            for rows in self.permuteRows():
                out.append(rows)
        self.insert(saveGrid)
        return iter(out)


    def fullPermutation(self) -> iter[Tuple[V]]:
        """ performs a full permutation of a grid, @see permuteGrid(),
            adding for each grid the respective diagonal reflection, @see diaflect(),
            resulting in 3.359.232 grid permutations for BASE_NUMBER == 3!!!
            
            NB: this will take a while ... (ca. 7-8 min)!!!
        """
        out: List[Tuple[V]] = []
        grids: iter[Tuple[V]] = self.permuteGrid()
        while True:
            try:
                grd: Tuple[V] = next(grids)
                out.append(grd)
                self.insert(grd)
                out.append(self.gridOut(coord=col))
            except StopIteration:
                return iter(out)
        


    """ G.    TRANSLATION  """


    # TODO:  assert --> Exception! 
    def translate(self, gridTuple: Optional[Tuple[V]] = None, 
                  insert: bool = True) -> Optional[Tuple[V]]:
        """
        'Translates' the grid values from integer to a corresponding alphabtic 
        representation (= "abcGrid"), or vice versa. 
        The abcGrid is initialized alphabetically, meaning that translation
        from any grid will always have [A, B, C, D, E, F, G, H, I]" in row 1. 
        As a consequence, back-translation to int will have [1, 2,3, 4, 5, 6, 7, 8, 9]
        in row 1
        (it is still the same "deep grid" -- just differently encoded; 
         the original number distribution can be restored via method@recode). 

        Parameters
        ----------
        gridTuple : Optional[Tuple[V]], optional
            DESCRIPTION. The default is None.
        insert : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        Optional[Tuple[V]]
            DESCRIPTION.

        """
        if gridTuple is None:
            gridTuple = self.gridOut()

        assert len(set(gridTuple)) != 0, "Grid not initialized!"
        assert issubclass(type(gridTuple[0]), (str, int)) 
        sourceType: type = type(gridTuple[0])        
        assert [type(x) is sourceType and str(x).isalnum() 
                for x in gridTuple]
    
        encoder: Dict[V, V]
        if sourceType is int:
            ABC_START_Idx = 65
            abc: list[str] = [chr(ABC_START_Idx + x) 
                              for x in range(self.DIMENSION)]
            encoder = {gridTuple[idx] : abc[idx] 
                   for idx in range(len(abc))}
        else:       # sourceType is str:
            encoder = {gridTuple[idxVal-1] : idxVal 
                   for idxVal in range(1, self.DIMENSION + 1)}
        
        out: List[V] = []
        for key in gridTuple:
            out.append(encoder[key])
        if not insert:
            return tuple(out)
        self.insert(out)



    def recode(self, encode: Collection[V], gridTuple: Optional[Tuple[V]] = None,
                  insert: bool = True) -> Optional[Tuple[V]]:
        """
        'translates' a given grid via the key {encode}, which must be a valid 
        Sudoku sequence of size {DIMENSION} (= 9 for classical Sudoku). 
        The key substitutes the first row and forms the basis for an encoder 
        with gridRow_1[n] -> encode[n]; for example:
            
            given grid row 1:   [1, 2, 3, 4, 5, 6, 7, 8, 9]
            encode:             [9, 8, 7, 6, 5, 4, 3, 2, 1]
            
          ==>  newEncoder:      1 -> 9
                                2 -> 8
                                3 -> 7
                                4 -> 6
                                5 -> 5
                                6 -> 4
                                7 -> 3
                                8 -> 2
                                9 -> 1
        
        Finally, the grid values are substituted by this encoder.
        Works from int to int, int to str, str to str, and str to int. 

        Parameters
        ----------
        encode : Collection[V]
            DESCRIPTION.
        gridTuple : Optional[Tuple[V]], optional
            DESCRIPTION. The default is None.
        insert : bool, optional
            DESCRIPTION. The default is True.

        Returns
        -------
        Optional[Tuple[V]]
            DESCRIPTION.

        """
        if gridTuple is None:
            gridTuple = self.gridOut()
        assert len(set(encode)) == self.DIMENSION, \
        f"{self.DIMENSION} keys required; {len(set(encode))} were given!"
        assert not (0 in gridTuple), "Grid not initialized!"
        assert issubclass(type(gridTuple[0]), (str, int)) 
        sourceType: type = type(gridTuple[0])        
        assert [type(x) is sourceType and str(x).isalnum() 
                for x in gridTuple]
        newEncoder: Dict[V, V] = {gridTuple[idx] : encode[idx] 
                                  for idx in range(len(encode))}
        out: List[V] = []
        for key in gridTuple:
            out.append(newEncoder[key])
        if not insert:
            return tuple(out)
        self.insert(out)





    def reduceGrid(self, gridTuple: Optional[Tuple[V]] = None, typ: type = str) -> V: 
        """
        Turns some grid content -- tuple of digits/characters -- into a narrow
        sequence, by default a string 
        (2, 7, 4, 9, 1, 3, 5, 6, 8, 1, 5 ...)   --> "27491356815 ...", or 
        ('A', 'B', 'C', 'D', 'E', 'F', 'G ...)  --> "'ABCDEFGHIEG ...". 
         The optional setting typ=int works only for numeric grids: 
        (2, 7, 4, 9, 1, 3, 5, 6, 8, 1, 5 ...)   --> 27491356815 ... (type: int). 


        Parameters
        ----------
        grid : TYPE:        boolean;  optional
            DESCRIPTION:    
                            The default is False.
        typ : TYPE, optional
            DESCRIPTION. The default is str.

        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        out = ""
        if not gridTuple:
            gridTuple = self.gridOut()
        for val in gridTuple:
            out += str(val)
        return typ(out)






    def collectReducedGrids(self, gridColl: Tuple[Tuple[V]], typ: type = str) -> Tuple[V]:
        """
        Compresses a collection of grid-tuples to a tuple of grid-strings.

        Parameters
        ----------
        gridColl : Tuple[Tuple[V]]
            DESCRIPTION.
        typ : type, optional
            DESCRIPTION. The default is str.

        Returns
        -------
        Tuple[V]
            DESCRIPTION.

        """
        out: List[Tuple[V]] = []
        for grid in gridColl:
            out.append(self.reduceGrid(grid, typ))
        return tuple(out)





    def doTheABC(self) -> iter[str]:
        """
        Creates a grid collection by generating a full permutation series (method@fullPermutation),
        and 'translates' into a collection of abcGrids.  

        NB: this will take a while ... (ca. 10 min)!!!

        Raises
        ------
        Exception
            DESCRIPTION.

        Returns
        -------
        iter[str]
            collection of abcGrids.

        """
        if self.gridOut() != self.zeroSeq:
            gridColl: iter[Tuple[V]] = self.fullPermutation()
        else:
            raise Exception("No Permutation of ZeroGrids!")
        out: List[str] = []
        while True:
            try: 
                grid: Tuple[V] = next(gridColl)
                grid = self.translate(grid, insert=False)
                out.append(grid)
            except StopIteration:
                return iter(self.collectReducedGrids(out)) 
            




# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             

    def saveGridCollection(self, gridColl: iter[V], gridNr: int, name: str = "ABC_Perm_") -> None:
        """
        A shortcut for saving a (reduced) grid collection to a .txt file; 
        default name: 'ABC_Perm_{gridNr}.txt'   (make sure that {gridNr} is still available)

        Parameters
        ----------
        gridColl : iter[str]
            collection of grids; 
            it is advised that .doTheABC has performed on the collection first.
        gridNr : int
            running number of the filename.

        Returns
        -------
        None.

        """
        import json
        outCollection = tuple(gridColl)
        file = name + str(gridNr) + ".txt"
        with open(file, "w") as f:
            json.dump(outCollection, f)

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             




# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             




C = TypeVar('C', Tuple, iter) 

# TODO! ... work in progress, so there are several todos 
class GridCollection:
    """
        This class provides a data structure to process collections 
        of sudoku grids, in particular permutation series produced by 
        some Grid object. 
        . . . . . .
        . . . . . .
        This structure itself is a pre - neural-network architecture, but intended
        to perform similar operations, e.g. find similar patterns, validity,
        symmetries etc. across a large number of grids (= grid tuples). 
        It will, however, also be possible to produce training/validation 
        and test sets to be processed in an actual NN. 
    """
    def __init__(self, collection: Optional[C] = None, file="ABC_Perm_0.txt"):
        import json
        import collections.abc
        
        self.size: int
        self.BASE_NUMBER: int 
        self.DIMENSION: int 
        self.size_permutations: int = 0
        
        self._INT_collection: Tuple[Tuple[int]]

        self.__input: iter 

        self._straightCollection: Tuple[Tuple[int]] 
        self._permutationCollection: Tuple[V] = None
        
        if collection:
            if isinstance(collection, collections.abc.Iterable):
                self.__input = collection
                collection = list(collection)
            if isinstance(collection, collections.abc.Collection):
                self.__input = iter(collection)
                self._unpack(collection)
        elif file:
            with open(file, "r") as f:
                collection = json.load(f)
            self.__input = iter(collection)
            self._unpack(collection)
        else:
            raise IOError("No input collection given!")
        
        
    def _unpack(self, inputList: List[V]) -> None:
        self.size = len(inputList)
        self.BASE_NUMBER = int(sqrt(sqrt(len(inputList[0]))))
        assert self.BASE_NUMBER**4 == len(inputList[0])
        assert len(inputList[0]) == len(inputList[1])
        self.DIMENSION = self.BASE_NUMBER**2
        grd: Grid = Grid(self.BASE_NUMBER)
                
        if type(inputList[0][0]) == str:
            intSequence: List[int] = [grd.translate(seq, insert=False) 
                                 for seq in inputList]
            self._INT_collection = intSequence
        elif type(inputList[0][0]) == int:
            self._INT_collection = inputList
        else:
            raise ValueError(f"Input type {type(inputList[0][0])} cannot be handled!")


    @property 
    def INT_collection(self):
        try:
            return self._INT_collection
        except AttributeError:
            raise Exception("INT_collection not instantiated!")
    
    @INT_collection.setter 
    def INT_collection(self, collection: Tuple[Tuple[int]]) -> None:
        self._INT_collection = collection
        
        



    # TODO
    def instantiate_PermutationSeries(self, idx: int):
        
        
        grd = Grid(self.BASE_NUMBER)
        startGrid: Tuple[int] = self.INT_collection[idx]
        
        
        
        tmp = list(range(1, self.BASE_NUMBER**2+1))
        keys: iter = permutations(tmp)
        out: List[Tuple[int]] = []
        while True:
            try: 
                key: Tuple[int] = next(keys)
                out.append(grd.recode(key, startGrid, insert=False))
            except StopIteration:
                self.RND_collection = tuple(out)
                self.size_permutations = len(out)
                return 


            
        

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             
# * * * * * * * * * * * * * *  KILLER TESTER  * * * * * * * * * * * * * * * * 
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             

class KillerSolver:
    """
        Data structure to test possible sums / combinations of sums 
        for killer sudoku. 
    """
    def __init__(self, BASE_NUMBER: int = 3) -> None:
        self.BASE_NUMBER: int = BASE_NUMBER
        self.MAX_SUM: int = self.addUp(self.BASE_NUMBER**2)
        self.range = tuple(range(1, BASE_NUMBER**2 + 1))
        self.currentSum: int = 0
        self.sums: list[tuple] = []
        
    def addUp(self, maxInt: int) -> int:
        sum: int = 0
        for i in range(1, maxInt + 1):
            sum += i 
        return sum
    
    
    def sumParser(self, summa: int, cells: int) -> tuple[tuple[int]]:
        combis = combinations(self.range, cells)
        out: list[tuple[int]]= []
        while True:
            try:
                summands: tuple[int] = next(combis)
                if sum(summands) == summa:
                    out.append(summands)
            except StopIteration:
                return tuple(out)
    
    def addNextSum(self, summa: int, cells: int) -> None:
        if self.currentSum + summa > self.MAX_SUM:
            raise Exception(f"The sum excedes the maximum {self.MAX_SUM} ({self.currentSum + summa})") 
        newSums: tuple[int] = self.sumParser(summa, cells)
        if len(newSums) == 0:
            raise Exception("The sum cannot be constructed!")
        self.currentSum += summa
        self.sums.append(newSums)
        
    
    def checkSums(self):
        if len(self.sums) == 0:
            return [self.range]
        if len(self.sums) == 1:
            return self.sums
        sums = product(*self.sums)        
        possibleSumCombis = []
        while True:
            try:
                possibleSums = next(sums)
                if len(possibleSums) == len(set(possibleSums)):
                    possibleSumCombis.append(possibleSums)
            except StopIteration:                
                return self.checkCompatibility(possibleSumCombis) 


    def checkCompatibility(self, sumCombis):
        out = []
        for sumCombi in sumCombis:
            tester = [item 
                 for oneSum in sumCombi 
                 for item in oneSum]
            if len(tester) == len(set(tester)):
                out.append(tuple(sumCombi))
        return tuple(out)



# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             
# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             



def generateRndGrid_cell(baseNumber: int, attempts: int = 1000, test_runs: int = 4000, 
                        prnt: bool = False) -> Tuple[int]:
    """
    Generates a grid assignment with a valid digit distribution at random.
    Sloppy version, does not use backtracking  
    --> can result in a quick output (or not), but it is possible
    that it does not produce any output at all because the procedure 
    >> runs out of options << as it were. Posssible fix:
        -- increase the {test_runs} parameter : how many grid permutations 
                                                are tested in one go?
        -- increase the {attempts} parameter  : how many permutation series
                                                are run?

    Parameters
    ----------
    attempts : int,  optional
        DESCRIPTION: Number of times the aux method is called; 
                     the default is 1000.
    test_runs : int, optional
        DESCRIPTION: Number of times the aux function may fail at inserting 
                     a random number into the grid.
                     The default is 4000.
    prnt : bool, optional:  if True: the solution will be printed on screen
        DESCRIPTION. The default is True.

    Returns
    -------
    None
        DESCRIPTION.

    """
    
    for i in range(attempts):
        testGrid: Grid = Grid(baseNumber)
        if _rnd_single(testGrid, test_runs = test_runs):
            if prnt == True:
                print(i * test_runs, " attempts!")
                testGrid.showGrid()
            return testGrid.gridOut()
    return tuple()

def _rnd_single(testGrid: Grid, test_runs: int) -> bool:
    """ @_generateRndGrid_aux """
    run: int = 0
    while run < (testGrid.BASE_NUMBER**4):
        row: int = testGrid.baseGrid[run].row
        col: int = testGrid.baseGrid[run].col
        box: int = testGrid.baseGrid[run].box
        val: int = randint(1, testGrid.DIMENSION)
        if val in testGrid.gridRow(row) or \
           val in testGrid.gridCol(col) or \
           val in testGrid.gridBox(box):
            test_runs -= 1                
        else:
            testGrid.baseGrid[run].val = val
            run += 1
        if test_runs < 0:
            return False
    return True

# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             

def generateRndGrid_deep(baseNumber: int) -> Tuple[int]:
    sequence: Tuple[int] = LinkedSequence(baseNumber)
    return sequence.sequence

class Node:
    def __init__(self, baseNumber: int, parent: Node, 
                 valSequence: Tuple[int] = []):
        self.level: int = 1 if not parent else parent.level + 1     ####
        self.parent: Node = parent
        self.child: Node = None 
        self.valueSequence: Tuple[int] = valSequence
        self.dimension: int = baseNumber**2
        self.sequenceSoFar = self._getCurrentSequence() 
        self.possibleValues: iter[Tuple[int]] = self._getValues()
        
        
    def _getValues(self) -> iter[Tuple[int]]: 
        possibleValues: List[int] = list(self.valueSequence) if self.valueSequence else list(range(1, self.dimension+1))
        for i in range(randint(0, 42)):
            shuffle(possibleValues)
        return permutations(possibleValues)

    def _getCurrentSequence(self) -> Tuple[int]:
        sequence: list[int] = []
        node: Node = self
        while node != None:
            tmpSeq = list(node.valueSequence)
            tmpSeq.reverse()
            _ = [sequence.insert(0, val) 
                 for val in tmpSeq]
            node = node.parent
        return tuple(sequence)

class LinkedSequence:
    def __init__(self, baseNumber: int):
        self.BASE_NUMBER = baseNumber
        self.DIMENSION = baseNumber**2
        
        rootValue: List[int] = list(range(1, self.DIMENSION+1))
        for shuffling in  range(randint(0, factorial(self.DIMENSION)-1)):
            shuffle(rootValue)
        self.sequence: Tuple[int] 
        self.root: Node = Node(self.BASE_NUMBER, parent=None, valSequence=rootValue)
        self.initializeNode(self.root) 

    
    def initializeNode(self, node: Node):
        print(f"initialize @Level 1:  {node.valueSequence}")
        while node.level < self.DIMENSION:      ####            
            node = self.nextNode(node)
            self.sequence = node.sequenceSoFar
        
    def nextNode(self, node: Node) -> Node:         
        testGrid: Grid = Grid(self.BASE_NUMBER) 
        while True:
            try:
                candidateSeq: tuple[int] = next(node.possibleValues)
                testSequence: List[int] = list(testGrid.zeroSeq)
                for idx in range(len(node.sequenceSoFar)):
                    val: int = node.sequenceSoFar[idx]
                    testSequence[idx] = val                 
                addIdx: int = testSequence.index(0)
                for idx in range(len(candidateSeq)):
                    testSequence[idx + addIdx] = candidateSeq[idx]
                testGrid.insert(testSequence)
                if testGrid.gridCheckZero():
                    nextKid: Node = Node(self.BASE_NUMBER, node, candidateSeq)
                    node.child = nextKid 
                    print(f"success    @Level {nextKid.level}:  {candidateSeq}")
                    return nextKid
            except StopIteration:
                return node.parent
                










# * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *             
# Test grids for illustration

grd1 = (7, 2, 4, 9, 1, 3, 5, 6, 8,
        5, 1, 9, 6, 8, 7, 3, 4, 2,
        3, 8, 6, 2, 5, 4, 1, 9, 7,
        2, 3, 1, 4, 7, 9, 6, 8, 5,
        4, 6, 7, 5, 3, 8, 2, 1, 9,
        8, 9, 5, 1, 6, 2, 7, 3, 4,
        1, 7, 8, 3, 4, 5, 9, 2, 6,
        9, 4, 3, 7, 2, 6, 8, 5, 1,
        6, 5, 2, 8, 9, 1, 4, 7, 3)

grd2 = (2, 7, 4, 9, 1, 3, 5, 6, 8,
        1, 5, 9, 6, 8, 7, 3, 4, 2,
        8, 3, 6, 2, 5, 4, 1, 9, 7,
        3, 2, 1, 4, 7, 9, 6, 8, 5,
        6, 4, 7, 5, 3, 8, 2, 1, 9,
        9, 8, 5, 1, 6, 2, 7, 3, 4,
        7, 1, 8, 3, 4, 5, 9, 2, 6,
        4, 9, 3, 7, 2, 6, 8, 5, 1,
        5, 6, 2, 8, 9, 1, 4, 7, 3)

grd3 = (3, 7, 8, 4, 2, 6, 5, 9, 1,
        2, 5, 4, 1, 8, 9, 6, 3, 7,
        1, 9, 6, 7, 3, 5, 4, 2, 8,
        5, 6, 9, 3, 7, 8, 2, 1, 4,
        7, 3, 2, 5, 1, 4, 8, 6, 9,
        4, 8, 1, 9, 6, 2, 7, 5, 3,
        9, 2, 5, 8, 4, 3, 1, 7, 6,
        8, 1, 3, 6, 5, 7, 9, 4, 2,
        6, 4, 7, 2, 9, 1, 3, 8, 5)

grd4 = (2, 6, 4, 3, 8, 9, 5, 1, 7,
        5, 1, 7, 6, 4, 2, 9, 8, 3,
        3, 8, 9, 7, 5, 1, 4, 6, 2,
        4, 2, 6, 5, 1, 7, 3, 9, 8,
        9, 3, 8, 2, 6, 4, 1, 7, 5,
        1, 7, 5, 8, 9, 3, 6, 2, 4,
        6, 4, 2, 1, 3, 8, 7, 5, 9,
        7, 5, 3, 9, 2, 6, 8, 4, 1,
        8, 9, 1, 4, 7, 5, 2, 3, 6)

grd5 =  (3, 1, 7, 2, 4, 6, 5, 9, 8, 
         5, 8, 6, 7, 3, 9, 2, 1, 4, 
         4, 9, 2, 1, 8, 5, 6, 7, 3, 
         9, 3, 4, 5, 2, 8, 1, 6, 7, 
         7, 2, 8, 6, 9, 1, 4, 3, 5, 
         6, 5, 1, 4, 7, 3, 9, 8, 2, 
         8, 6, 5, 3, 1, 4, 7, 2, 9, 
         1, 7, 3, 9, 5, 2, 8, 4, 6, 
         2, 4, 9, 8, 6, 7, 3, 5, 1)



            

if __name__ == '__main__': 
    print("====================================================")
    print("*                                                  *")
    print("*          The syntax of Sudoku Grids              *")
    print("*                                                  *")
    print("*            part 1: Infrastructure                *")
    print("*                Cells, Grids,                     *")
    print("*              and permutations                    *")
    print("*               (and much more)                    *")
    print("*                                                  *")
    print("====================================================")



