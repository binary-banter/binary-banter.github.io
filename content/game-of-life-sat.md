+++
title = "Sat Solving Game of Life"
date = 2023-11-14
draft = true

[extra]
image = "/assets/game-of-life/thumbnail.png"

[taxonomies]
tags = ["rust", "game-of-life", "TODOODODODODODOOOO"]
+++

Half a year ago we [showed](/game-of-life) how game of life can be parallised, first on the CPU using SIMD, and then on the GPU.
We discovered that CUDA has an instruction called <emph>lop3</emph>, which can encode an arbitrary 3-bit truth table, and applies that truth table to the bits of 3 registers. In this blogpost we (try to) find the minimum number of lop3 instructions needed to solve game of life, using a SAT solver.

<!-- more -->

# Table of Contents

<!-- TOC -->
TODO
<!-- TOC -->

# Quick reminder of the problem from previous blog

For a quick reminder, game of life is played rectangular grid populated by cells that can be either <emph>alive</emph> or <emph>dead</emph>.
The state of the cells are simultaneously updated in steps using the following rules:
1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
2) Any live cell with two or three neighbours lives on to the next generation.
3) Any live cell with more than three live neighbours dies, as if caused by over-population.
4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

The goal of this blog post is to find a set of [lop3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3) instructions such that given a cell and the 8 neighbouring cells, we can determine next state of that cell. Lop3 is an instruction that takes 3 inputs `a`, `b` and `c`, an output `d` and a look-up-table `LUT` which it uses to compute arbitrary bitwise logical operations.

Using our human ingenuity we managed to find a set of 11 `lop3` instructions that does exactly that:
```
a8        = lop3.b32     a2        a1 a0 0b10010110
b0        = lop3.b32     a2        a1 a0 0b11101000
a9        = lop3.b32     a5        a4 a3 0b10010110
b1        = lop3.b32     a5        a4 a3 0b11101000
       
aA        = lop3.b32     a8        a7 a6 0b10010110
b2        = lop3.b32     a8        a7 a6 0b11101000
b3        = lop3.b32     b2        b1 b0 0b10010110
c0        = lop3.b32     b2        b1 b0 0b11101000

parity    = lop3.b32 center        aA a9 0b11110110
two_three = lop3.b32     b3        aA a9 0b01111000
center    = lop3.b32 parity two_three c0 0b01000000
```

The code works as follows:
- The first 8 instructions are equivalent to four full-adders, similar to CPU code from the previous blog post.
  - `0b10010110` is a 3-bit XOR, producing the sum bit of the full-adder.
  - `0b11101000` is an at-least-two operation, producing the carry bit of the full-adder.
- These instructions have reduced the information down to the bits `c0`, `b3`, `a9`, and `aA`, which represent `count = 4*c0 + 2*b3 + (a9 + aA)`.
- `parity` has the following meaning:
  - If center is on, stay on
  - If the number of live neighbours is odd (determined from a9 and aA), turn on.
    This will make sure the cell turns on if it has 3 live neighbours.
    Note that this will also erroneously turn the cell on if 1, 5 or 7 neighbours are on, but in those cases the cell will be killed by the final instruction.
- `two_three` is on if the total value of `b2`, `a9`, and `aA` is 2 or 3. Remember that `b2` has a weight of 2.
- Finally, the cell is live if:
  - `parity` is on, meaning the cell is alive or the number of live neighbours is odd.
  - `two_three` is on, meaning the cell has two or three live neighbours.
  - `c0` is off, meaning the cell does not have more than 4 live neighbours.

This means that we can do a single step using only 11 `lop3` instructions. 
But who's to say that 11 instructions is the best solution? Maybe we can do 10? or 9? 

# Intro to SAT & CNF

A <emph>SAT solver</emph> can tell us! A SAT solver takes an arbitrary boolean formula, such as `(x | y) & (!x | !y)`, and it outputs one of:
* <emph>SAT</emph> if the problem is "satisfiable", meaning that it has a solution. It will then also output a solution, such as `x = 1, y = 0` to the problem above.
* <emph>UNSAT</emph> if the problem is "unsatisfiable", meaning that it has no solution. Some SAT solvers can then also produce a proof of impossibility. This is not relevant for this blogpost, but to see how this works, see [here](https://www.msoos.org/2022/04/proof-traces-for-sat-solvers/).

SAT solvers take their input in a very specific form, called <emph>CNF</emph> (Conjuctive Normal Form).
In CNF, variables are numbered starting from 1, and constraints are <emph>conjunctions</emph> of possibly negated variables.
For example `1 2 -3` is equivalent to `1 | 2 | !3`. Every boolean formula can be represented as a set of CNF constraints. For example, see the following transformations:
```
1 | 2    =>  (1 2)
1 & 2    =>  (1) (2)
1 -> 2   =>  (!1 2)
1 <-> 2  =>  (!1 2) (1 !2)
```

# Naive translation
We'll <emph>encode</emph> our game of life problem into a boolean problem.



- Decoding SAT solutions

# Symmetries / Prune
- Create an order

# Assumptions / Relaxation
- This can only prove SAT

# Solving subproblems 
- Explain how we vary inputs
- Kissat / Parkissat
- Excel sheet here + Add computation hours
- HPC and how much cpu power

# Future Work
- Z3
- Call to action


