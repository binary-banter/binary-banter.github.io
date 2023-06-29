+++
title = "Game of Life: How a nerdsnipe led to a fast implementation of game of life"
date = 2023-06-17

[taxonomies]
tags = ["rust", "game-of-life", "gpgpu", "simd"]
+++

Inspired by an Advent of Code problem, we spontaneously decided to explore Conway's Game of Life and discussed various techniques to solve it efficiently. 
This blogpost provides a concise summary of our weeks-long journey, where we delved into finding faster solutions for the Game of Life.

<!-- more -->

# Table of Contents

<!-- TOC -->
* [Table of Contents](#table-of-contents)
* [A Nerdsnipe](#a-nerdsnipe)
* [Trivial simulation](#trivial-simulation)
* [Unleashing the Power of Parallelism Using a Packed Representation](#unleashing-the-power-of-parallelism-using-a-packed-representation)
  * [A Packed Representation Using Bits](#a-packed-representation-using-bits)
  * [A Packed Representation Using Nibbles](#a-packed-representation-using-nibbles)
  * [Back to a Packed Representation Using Bits](#back-to-a-packed-representation-using-bits)
* [Embracing the Spectacle of Parallelism Using SIMD and Multi-Threading](#embracing-the-spectacle-of-parallelism-using-simd-and-multi-threading)
* [Ascending to the Apex of Parallelism Using the GPU](#ascending-to-the-apex-of-parallelism-using-the-gpu)
  * [OpenCL](#opencl)
  * [Cuda](#cuda)
* [Comparing Results with Previous Work](#comparing-results-with-previous-work)
<!-- TOC -->

# A Nerdsnipe

On a sunny afternoon, for no particular reason, we decided to discuss an [Advent of Code problem](https://adventofcode.com/2020/day/11) from a couple of years ago.
This problem was about a cellular automaton which was strongly inspired by Conway's Game of Life.
[Here](https://www.youtube.com/watch?v=C2vgICfQawE) is an <emph>epic</emph> video showing what it looks like.

Obviously we had to discuss how we would solve this problem in the fastest manner possible.
One thing let to another, and needless to say, this discussion went on to consume weeks of our lives as we tried to find faster and faster solutions.

This blogpost is a short summary of all the techniques we've tried to speed up Game of Life.

The source code for this project is available on [GitHub](https://github.com/binary-banter/fast-game-of-life/).

# Trivial simulation

Before we move on to explaining the techniques we developed, we will first give a short recap of how Game of Life works.
The game is played on a rectangular grid populated by cells that can be either <emph>alive</emph> or <emph>dead</emph>.
The state of the cells are simultaneously updated in steps using the following rules:
1) Any live cell with fewer than two live neighbours dies, as if caused by under-population.
2) Any live cell with two or three neighbours lives on to the next generation.
3) Any live cell with more than three live neighbours dies, as if caused by over-population.
4) Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

The simplest possible simulation of these rules works on a 2D array of booleans, literally following the rules above. 
For each cell the program evaluates its current state and number of live neighbours, which it uses to compute the next state to be stored in a secondary grid.
The secondary grid is necessary because otherwise some cells might be using neighbours from the *next* step.
Assuming booleans are stored as bytes, we store 1 cell per byte and need 9 memory accesses to determine its next state (1 for the center and 8 for its neighbours).

At the end of the step, the two grids are swapped either inefficiently by copying the entire secondary grid back to the primary grid, or more efficiently by swapping their pointers.

# Unleashing the Power of Parallelism Using a Packed Representation

The naive solution clearly doesn't make optimal use of memory. 
We not only need an entire byte to store something that essentially only needs 1 bit to indicate whether it is *dead* or *alive*, 
but we also need many memory accesses on top of that for just a single cell.

## A Packed Representation Using Bits
Our first idea to improve upon this solution was to use a <emph>packed representation</emph>, where each bit represents a cell.
It would then be possible to retrieve these cells by using a mask and shifts, making this solution slower than our original solution.

## A Packed Representation Using Nibbles
We can achieve a packed representation that is faster than our original by harnessing the power of parallelism.
To make our lives easier for now, we store each cell in a <emph>nibble</emph> (4 bits).
We store these nibbles in an array of `u64`s, which we will call <emph>columns</emph>.
Cells can then be stored in a column like this:
* 1001 1011 0000 0011 -> 0x1001_1011_0000_0011

Our goal is to find a way to simulate these 16 cells in parallel.
Note that to count the neighbours of a cell, we could simply add their corresponding nibbles.
The difficulty lies in shifting the neighbours around in such a way that they align, so that we can add them.

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


In pseudocode, we can find the count of each cell as follows:

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

## Back to a Packed Representation Using Bits

So far we have stored a cell in each nibble, but we can actually implement a parallel simulator that stores a cell in each bit. 
This makes the logic a lot more complex to understand, but it does increase throughput.

The addition of the 8 neighbouring cells can be written as:
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

Luckily we recently took a course on Computer Arithmetic, which taught us exactly how to do this! 
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

# Embracing the Spectacle of Parallelism Using SIMD and Multi-Threading

The story doesn't end there however!
We have merely used parallelism on the scale of `u64`s, but most modern CPUs actually provide specialized instructions that work on registers of multiple `u64`s.
These instructions are commonly referred to as Single-Instruction Multiple-Data, or <emph>SIMD</emph> for short.

We use Rust's [new nightly std::simd API](https://doc.rust-lang.org/std/simd/index.html) to achieve this. 
The API provides easy to use const-generic methods that abstract away the underlying intrinsics.

For the most part the translation from `u64`s as columns to `N` `u64`s as columns, where `N` is the number of SIMD lanes, was quite straight forward.
The only real hurdle we came across is the lack of register-wide bit-shifts.
Instead of shifting the entire register, the lanes themselves are shifted.

To remedy this problem we implemented helper functions. Below is our shift_left function.
```rs
pub fn shl_1<const N: usize>(v: Simd<u64, N>) -> Simd<u64, N>
where
    LaneCount<N>: SupportedLaneCount,
{
    let approx = (v << Simd::splat(1));

    let mut mask = [0x00000_0000_0000_0001; N];
    mask[N - 1] = 0;

    let neighbouring_bits = (v >> Simd::splat(63)).rotate_lanes_left::<1>() & Simd::from_array(mask);
    approx | neighbouring_bits
}
```
The function works as follows:
* First we approximate the shift, by shifting each lane left by 1. This works for all bits except the bits that should be shifted across lane boundaries.
* To get the remaining bits, we rotate the lanes right by one, then shift each lane left by 63. 
* We mask the result of this so only the relevant bits are selected. Specifically, the lane that was rotated around the register should be ignored.
* Finally, we `OR` these results together.

With the hardware available to us, we can now simulate 256 cells in parallel! 
However, this is not where the parallelism stops, since CPUs have multiple cores, we can take advantage of <emph>multi-threading</emph>.
During each step, we can split the rows of the field into `N` equal-sized groups, each of which is simulated by a separate thread. 
We use a thread pool so the overhead of spawning the threads is amortized over the steps.

While initially we thought that the Rust ownership model would hinder us when writing the results to the same array from multiple threads, Rust has a beautiful abstraction called `chunks_mut` that does exactly what we want.
This splits a slice into `N` equal-sized groups. The rest of this change was trivial, requiring only a few changes in the way we index the arrays.

Now we are fully using our CPU! But this is not the end of our journey...

# Ascending to the Apex of Parallelism Using the GPU

We have now seen how we can efficiently pack cells into different datastructures and quickly calculate the next state for each cell.
At the end of the last chapter we made our solution multi-threaded. Modern CPUs have many cores that can perform work at the same time, but GPUs actually have *thousands* of cores that can perform simple tasks at the same time.
GPUs however have a fastly different model of computation and memory, so running our solution on the GPU is not trivial.
Yet still we were determined to take a look into General Purpose GPU (<emph>GPGPU</emph>) programming.

There are two major frameworks for doing GPGPU: OpenCL and CUDA.
Since CUDA is only available for NVidia GPUs, we decided to first focus on porting our solution to OpenCL.

## OpenCL

We decided to use the [opencl3](https://crates.io/crates/opencl3/) crate for our OpenCL implementation, since it provides a simple interface for working with OpenCL.
OpenCL makes a distinction between the <emph>host</emph> (CPU) and <emph>device</emph> (usually GPU).

There are some key differences when comparing writing code for a CPU versus writing OpenCL code.
In particular, the memory of the device is usually separate from that of the host, which means explicit calls need to be made to move data back and forth.
Furthermore, kernels (i.e. programs) need to be compiled and loaded into the device which are then all ran in parallel.

Our first implementation was quite straightforward - a kernel that simulates a column for a single step with the same logic used on the CPU.
During the step the kernel will load its neighbouring columns from *global* memory.
This kernel takes two buffers of 32-bit unsigned integers `field` and `new_field`, which are swapped after the step has been computed for all columns.
By repeatedly launching this kernel with the global work size equal to the dimensions of the grid any number of steps can be simulated.

This basic implementation was already quite fast, but this was definitely not the fastest it could go.
We started experimenting using different techniques and benchmarked them using [criterion](https://crates.io/crates/criterion) in order to determine what works and doesn't work.
In the following subsections we will discuss the techniques that worked:
* Multi-step simulation
* Use shared memory (also called local memory)
* Increase work-per-thread (WPT)

### Multi-Step Simulation

### Shared Memory

### Work-Per-Thread

[//]: # (Optimizations:)
[//]: # (- Driven by benchmarks!)
[//]: # (- Multistep &#40;requires simulating extra cells, discuss trade-off&#41;)
[//]: # (- Shared memory to store & communicate intermediate results for multistep)
[//]: # (- Discuss different layouts we tried)
[//]: # (  - Final: Load 3 columns, simulate 2 using 16 bits)
[//]: # (  - Simulating 3 columns is not worth it &#40;show wasted simulation calculation&#41;)
[//]: # (- Work per thread)

## Cuda

OpenCL is a beautiful abstraction that allows you to write portable code that runs on a CPU, Amd GPU, Nvidia GPU, FPGA and more. 
This is an amazing feat given how different these architectures are, but this does mean that some of the details are abstracted away. 
Details that, if used properly, could allow for *more optimizations*!

### LOP3

One of these details is knowledge about the CUDA PTX instruction set and control over the instructions that are generated. 
We noticed while taking a look at our code using [Godbolt](https://godbolt.org/), 
that the compiler sometimes used an instruction called [lop3](https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#logic-and-shift-instructions-lop3) (lookup-3). 
What we discovered is that this is an *amazing* instruction!
It is an instruction that takes 3 inputs `a`, `b` and `c`, an output `d` and a look-up-table `LUT` which it uses to compute arbitrary bitwise logical operations.
The following table shows how the `LUT = 0b1001_0110` encodes the ternary bitwise XOR operator for inputs `a`, `b` and `c`.

| a | b | c | d |
|---|---|---|---|
| 0 | 0 | 0 | 0 |
| 0 | 0 | 1 | 1 |
| 0 | 1 | 0 | 1 |
| 0 | 1 | 1 | 0 |
| 1 | 0 | 0 | 1 |
| 1 | 0 | 1 | 0 |
| 1 | 1 | 0 | 0 |
| 1 | 1 | 1 | 1 |

Our step operation is a function that takes 9 integers as input (the 8 neighbours and the cell itself), and produces a new integer by looking at each 9-tuple of bits.
This means that if we can build our 9-input step function using 3-input lop3 instructions, it could potentially be a lot faster.

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
- The first 8 instructions are equivalent to the full-adder stages in the CPU code. 
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

This means that we can do a single step using only 11 `lop3` instructions, and the instructions necessary to construct the `a0`-`a7` neighbours, which are still the same as before.
However, we are not done yet! This is the solution that we as humans could come up with, but we gave the problem to a SAT-solver which found a solution with 10 `lop3` instructions.
The first 6 instructions are the same as ours, but the last 4 are complete magic to us, if you can understand what is happening here we're happy to hear it, but for now we've just accepted that a SAT-solver has beaten us.

```
// reduction stage
a8        = lop3.b32     a2     a1     a0 0b10010110
b0        = lop3.b32     a2     a1     a0 0b11101000
a9        = lop3.b32     a5     a4     a3 0b10010110
b1        = lop3.b32     a5     a4     a3 0b11101000
aA        = lop3.b32     a8     a7     a6 0b10010110
b2        = lop3.b32     a8     a7     a6 0b11101000

// magic stage dreamt up by an insane SAT-solver
magic0    = lop3.b32     a9     aA center 0b00111110
magic1    = lop3.b32 magic0 center     b2 0b01011011
magic2    = lop3.b32 magic1     b1     b0 0b10010001
center    = lop3.b32 magic2 magic0 magic1 0b01011000
```

A blog post about how we managed to translate this problem (and others) to a form that a SAT-solver can work with is coming soon.

### Warp Shuffle

The next observation we made is that the way we use shared memory is very regular, work items write to shared memory after each step, and at the start of the next step work items read the results of only their neighbouring work items.
This is something that can be efficiently encoded using a <emph>warp shuffle</emph> instead. A warp is a set of upto 32 work items, which can send information to each other using a warp shuffle. 

At the start of each step, each work item sends the state of the cells it is simulating to each neighbour, while simultaneously receiving the state of the neighbouring cells.
This completely eliminates the need for shared memory, allowing us to make use of the very fast register memory. 
One drawback of this approach is that we now have a maximum workgroup size of 32, while earlier we were able to run with a workgroup size of 512. 
This is not as drastic as it may seem, since each workgroup is split by the GPU into warps anyway, there is only some extra overhead in creating the extra workgroups.
Furthermore, due to the earlier technique of increasing the work-per-thread, the percentage of effective computation remains approximately the same.

```c
left[0] = __shfl_up_sync(-1, left[WORK_PER_THREAD], 1);
right[0] = __shfl_up_sync(-1, right[WORK_PER_THREAD], 1);
left[WORK_PER_THREAD + 1] = __shfl_down_sync(-1, left[1], 1);
right[WORK_PER_THREAD + 1] = __shfl_down_sync(-1, right[1], 1);
```

# Comparing Results with Previous Work

So how fast did we actually make it go? And was it significant?
Thanks to our awesome university, we were allowed to use the high performance cluster (HPC) to experiment with several GPUs.
We measure our performance in cell updates per second (CUpS). 

<table>
    <thead>
        <tr>
            <th>Implementation</th>
            <th>Hardware</th>
            <th>Performance (CUpS)</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td rowspan=3>Trivial</td>
            <td>i7 5820K</td>
            <td>0.219×10^9</td>
        </tr>
        <tr>
            <td>i7 6700K</td>
            <td>0.282×10^9</td>
        </tr>
        <tr>
            <td>i9 11900K</td>
            <td>0.378x10^9</td>
        </tr>
        <tr>
            <td rowspan=2>Packed Nibbles</td>
            <td>i7 5820K</td>
            <td>4.393×10^9</td>
        </tr>
        <tr>
            <td>i7 6700K</td>
            <td>5.065×10^9</td>
        </tr>
        <tr>
            <td rowspan=2>Packed Bits</td>
            <td>i7 5820K</td>
            <td>6.539×10^9</td>
        </tr>
        <tr>
            <td>i7 6700K</td>
            <td>7.033×10^9</td>
        </tr>
        <tr>
            <td rowspan=3>SIMD</td>
            <td>i7 5820K</td>
            <td>22.600x10^9</td>
        </tr>
        <tr>
            <td>i7 6700K</td>
            <td>27.875x10^9</td>
        </tr>
        <tr>
            <td>i9 11900K</td>
            <td>48.266x10^9</td>
        </tr>
        <tr>
            <td rowspan=3>SIMD Multi-threaded</td>
            <td>i7 5820K</td>
            <td>45.813x10^9</td>
        </tr>
        <tr>
            <td>i7 6700K</td>
            <td>71.105x10^9</td>
        </tr>
        <tr>
            <td>i9 11900K</td>
            <td>296.019x10^9</td>
        </tr>
        <tr style="height: 40px"></tr>
        <tr>
            <td rowspan=6>OpenCL</td>
            <td>1050 Notebook</td>
            <td>0.760×10^12</td>
        </tr>
        <tr>
            <td>1080Ti</td>
            <td>4.775×10^12</td>
        </tr>
        <tr>
            <td>V100</td>
            <td>6.678×10^12</td>
        </tr>
        <tr>
            <td>2080Ti</td>
            <td>6.783×10^12</td>
        </tr>
        <tr>
            <td>A40</td>
            <td>8.055×10^12</td>
        </tr>
        <tr>
            <td>7900XTX</td>
            <td>8.939×10^12</td>
        </tr>
        <tr>
            <td rowspan=5>CUDA</td>
            <td>1050 Notebook</td>
            <td>1.068×10^12</td>
        </tr>
        <tr>
            <td>1080Ti</td>
            <td>5.389×10^12</td>
        </tr>
       <tr>
            <td>V100</td>
            <td>9.871×10^12</td>
        </tr>
       <tr>
            <td>2080Ti</td>
            <td>9.666×10^12</td>
        </tr>
       <tr>
            <td>A40</td>
            <td>11.720x10^12</td>
        </tr>
    </tbody>
</table>

From previous work using Parallel Bulk Computation:

| Implementation | Hardware  | Performance (CUpS) |
|:--------------:|:---------:|:------------------:|
|      CPU       |   "i7"    |     13.4×10^9      |
|     GPGPU      |  Titan X  |    1.350×10^12     |
