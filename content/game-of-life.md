+++
title = "Game of Life: How a nerdsnipe led to a faster implementation of game of life"
date = 2023-06-17

[taxonomies]
tags = ["rust", "game-of-life"]
+++

# A Nerdsnipe

On a sunny afternoon, for no particular reason, we decided to discuss an [Advent of Code problem](https://adventofcode.com/2020/day/11) from a couple years ago.
This problem was about a cellular automaton, which was strongly inspired by Conway's Game of Life.
[Here](https://www.youtube.com/watch?v=C2vgICfQawE) is an <emph>epic</emph> video showing what it looks like.

Obviously we had to discuss how we would solve this problem in the fastest manner possible.
One thing let to another, and needless to say, this discussion went on to consume weeks of our lives as we tried to find faster and faster solutions.

This blogpost is a short summary of all the techniques we've tried to speed up Game of Life.

# Trivial simulation

Before we move on to explaining the techniques we developed, we will first give a short recap of how Game of Life works.
The game is played on a rectangular grid populated with cells that can be either <emph>alive</emph> or <emph>dead</emph>.
On each step the cell changes state based on the following rules:
1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
2) Any live cell with two or three neighbours lives on to the next generation.
3) Any live cell with more than three live neighbours dies, as if caused by over-population.
4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

The simplest possible simulation of these rules works on a 2D array of booleans, literally following the rules above. 
For each cell it examines its current state and counts the number of live neighbours, and it stores its next state in a secondary grid. 
This is necessary since the cells neighbours might not have been processed yet, and they still need to use the old state of the cell.
Assuming booleans are stored as bytes, we store 1 cell per byte and need 9 memory accesses to determine its next state (1 for the center and 8 for its neighbours).

At the end of the step, the two grids are swapped either inefficiently by copying the entire secondary grid back to the primary grid, or more efficiently by swapping their pointers.

# Unleashing the Power of Parallelism Using a Packed Representation

The naive solution clearly doesn't make optimal use of memory. 
We not only need an entire byte to store something that essentially only needs 1 bit for *dead* or *alive*, 
but we also need many memory accesses on top of that for just a single cell.

## A Packed Representation Using Bits
Our first idea to improve upon this solution was to use a <emph>packed representation</emph>, where each bit represents a cell.
It would then be possible to retrieve these cells by using a mask and shifts, making this solution slower than our original solution.

## A Packed Representation Using Nibbles
We can achieve a packed representation that is faster than our original by harnessing the power of parallelism.
To make our lives easier for now we store each cell in a <emph>nibble</emph> (4 bits).
We store these nibbles in an array of `u64`s, which we will call <emph>columns</emph>.
Cells can then be stored in a column like this:
* 1001 1011 0000 0011 -> 0x1001_1011_0000_0011

Our goal is to find a way to simulate these 16 cells in parallel.
Note that to count the neighbours of a cell, we could simply add their corresponding nibbles.
The hard part is shifting the neighbours around in such a way that they align, so that we can add them.

To get the top and bottom neighbours of a cell, we can simply load the columns above and below the cell and add them. 
The neighbours on the sides are more tricky, shifting the columns left and right almost does the trick, but leaves out one nibble on either side that needs to be loaded from the neighbouring column.
The following two tables show the cells 0 through 7 that we want to align with the center node `c`, so we can add them.
The second table highlights the case where some of the cells might be in the column to the left or right.

| left                | middle              | right               |
|---------------------|---------------------|---------------------|
| xxxx xxxx xxxx xxxx | xxxx x012 xxxx xxxx | xxxx xxxx xxxx xxxx |
| xxxx xxxx xxxx xxxx | xxxx x3c4 xxxx xxxx | xxxx xxxx xxxx xxxx |
| xxxx xxxx xxxx xxxx | xxxx x567 xxxx xxxx | xxxx xxxx xxxx xxxx |

| left                | middle              | right               |
|---------------------|---------------------|---------------------|
| xxxx xxxx xxxx xxx0 | 12xx xxxx xxxx xxxx | xxxx xxxx xxxx xxxx |
| xxxx xxxx xxxx xxx3 | c4xx xxxx xxxx xxxx | xxxx xxxx xxxx xxxx |
| xxxx xxxx xxxx xxx5 | 67xx xxxx xxxx xxxx | xxxx xxxx xxxx xxxx |


In pseudocode we can find the count of each cell as follows:

```rs
top          = grid[i - ROW]
bottom       = grid[i + ROW]

// Note that we shift with 60 = 15 * 4 to align the columns to the left and right.
left         = (grid[i] >> 4) | (grid[i - 1] << 60)
right        = (grid[i] >> 4) | (grid[i - 1] << 60)

top_left     = (grid[i - ROW] >> 4) | (grid[i - ROW - 1] << 60)
top_right    = (grid[i - ROW] << 4) | (grid[i - ROW + 1] >> 60)
bottom_left  = (grid[i + ROW] >> 4) | (grid[i + ROW - 1] << 60)
bottom_right = (grid[i + ROW] << 4) | (grid[i + ROW + 1] >> 60)
```

After aligning the cells we can now simply add them together to find the number of neighbours for each cell.
The following table show a grid of cells where 1 means alive and 0 dead.
The bottom rows shows the number of neighbours for each cell for the central column.

|         left         |        middle        |        right         |
|:--------------------:|:--------------------:|:--------------------:|
| xxxx xxxx xxxx xxx0  | 1001 1011 1000 0011  | 1xxx xxxx xxxx xxxx  |
| xxxx xxxx xxxx xxx1  | 1111 1000 0010 0110  | 1xxx xxxx xxxx xxxx  |
| xxxx xxxx xxxx xxx0  | 1001 1000 1100 0011  | 1xxx xxxx xxxx xxxx  |
| -------------------  | -------------------  | -------------------  |
|                      | 4446 5424 4411 1358  |                      |

The final challenge is to use the current state and these neighbour counts to compute the next state of the cells in parallel.
The exact instructions may seem a bit weird, but they produce the same results as the rules laid out earlier.

```rs
// start with the previous state
result  = grid[i]

// if we have 1,3,5,7 live neighbours, we are born
result |= count

// if we have 2,3,6,7 live neighbours, we survive
result &= count >> 1

// if we have 0,1,2,3,8 live neighbours, we survive
result &= !(count >> 2)

// if we have 0-7 live neighbours, we survive
result &= !(count >> 3)

// clear all bits that don't represent a cell
result &= 0x1111_1111_1111_1111
```

## Improving our packed representation

So far we have stored a cell in each nibble, but we can actually implement a parallel simulator that stores a cell in each bit. 
This makes the logic a lot more complex to understand for us mortals, but it does work a lot faster.

The addition we do above can be written down as:
```
   a
   b
   c
   d
   e
   f
   g
   h
---- +
qrst
```
Where `t` is the least-significant bit (LSB) of the `count`, 's' the LSB of 'count >> 1', etc.
We need to find a way to produce `q`, `r`, `s`, and `t` using bit-level instructions (AND, OR, NOT, XOR) instead of using the ADD instruction, so that we can simulate all 64 cells in parallel.

Luckily we recently took a course in computer arithmetic at TU Delft, which taught us exactly how to do this! 
What we need to build is called an <emph>8-bit counter</emph> as explained in [Parhami's Computer Arithmetic book](https://ashkanyeganeh.com/wp-content/uploads/2020/03/computer-arithmetic-algorithms-2nd-edition-Behrooz-Parhami.pdf).
Technically this method only really works well for hardware, but for this problem it actually works just as well in software!

We designed the following counter, where square brackets indicate full- and half-adders which are to be emulated in software.
The important thing to take away is that we reduce the 8 neighbours down to `count = 4x + 2(y+z) + w`.
```
      [a|
      |b|
      |c]
      [d|
      |e|
      |f]
      [g|
      |h]
--------- +
   [l|[i|
   |m||j|
   |n]|k]
--------- +
 x  y  w
    z
```

To implement the above table in software, we first implement full- and half-adders in software:
```rs
fn half_adder(a,b):
    sum = a ^ b
    carry = a & b
    return carry, sum

fn full_adder(a,b,c):
    temp = a ^ b
    sum = temp ^ c
    carry = (a & b) | (temp & c)
    return carry, sum
```

Next, we use the adders to implement the table above:   
```rs 
// stage 0
l, i = full_adder(a,b,c)
m, j = full_adder(d,e,f)
n, k = half_adder(g,h)

// stage 1
y, w = full_adder(i,j,k)
x, z = full_adder(l,m,n)
```

Finally, we use the results of the addition to update result:
```rs
// start with the previous state
result  = grid[i]

// if we have 1,3,5,7 live neighbours, we are born
result |= w;

// if we have 2,3,6,7 live neighbours, we survive
center &= (y ^ z);

// if we have 0,1,2,3 live neighbours, we survive
center &= ~x;
```

So we can now use this to simulate a fully packed `u64` of cells in parallel.

# Embracing the Spectacle of Parallelism Using SIMD

- TODO met simd

# Ascending to the Apex of Parallelism Using the GPU

- TODO met gpu