+++
title = "Building a Redstone Simulator in Rust"
date = 2023-05-25

[taxonomies]
tags = ["rust", "minecraft", "redstone"]
+++

# Table of Contents

<!-- TOC -->
* [Table of Contents](#table-of-contents)
* [Why we do what we do](#why-we-do-what-we-do)
* [A redstone simulator](#a-redstone-simulator)
* [How does Redstone work anyway?](#how-does-redstone-work-anyway)
  * [Weak and Strong Power](#weak-and-strong-power)
  * [Redstone Wire](#redstone-wire)
  * [Solid Blocks](#solid-blocks)
  * [Redstone Blocks](#redstone-blocks)
  * [Repeaters](#repeaters)
  * [Torches](#torches)
  * [Comparators](#comparators)
  * [Debouncing](#debouncing)
* [Simulating the world](#simulating-the-world)
  * [Direct simulation](#direct-simulation)
  * [Blocks](#blocks)
  * [Simulating the world as a graph](#simulating-the-world-as-a-graph)
  * [Creating a graph](#creating-a-graph)
  * [Pruning the graph](#pruning-the-graph)
* [Testing, lots of testing](#testing-lots-of-testing)
* [Visualizations](#visualizations)
* [Other stuff](#other-stuff)
* [Coding experience](#coding-experience)
<!-- TOC -->

<br />

# Why we do what we do

Recently, we decided to delve back into playing some Minecraft, but maybe not in the way you might be accustomed to.
Instead of playing the familiar vanilla survival mode, we set out to put Minecraft (and ourselves) to the test by
building a fully fledged 8-bit CPU using the game's redstone blocks. If you're curious, you can see the most recent
version of our CPU in the image below.

As we made progress and laid the foundation for our CPU, we realized the need to determine the worst-case response times
before proceeding with additional functionality. However, due to the growing complexity of our contraption, it became
increasingly difficult to identify the specific areas where delays were most likely to occur. To address this issue, we
came up with the idea of using external software to simulate our CPU. This approach would enable us to analyze signal
behavior and probe for potential bottlenecks more easily. And so, the idea for our Redstone Simulator was born.

In this blog, we will delve into the development and functionality of our Redstone Simulator, exploring the difficulties
and curiosities we came across along the way. Stay tuned for an exciting journey into the world of Minecraft's digital
circuitry!

The source code for this project is available on [GitHub](https://github.com/JonathanBrouwer/redstone-simulator) and was
written for Minecraft version 1.19.

{{ image(src="assets/cpu.png", alt="CPU in Minecraft", position="left", style="border-radius: 8px;") }}

<br />

# A redstone simulator

So with the idea of a Redstone Simulator in mind, we wanted to clearly define the scope of this project.
Importantly, We set the following goals for our project:

* Allow fine-grained probing of redstone contraptions.
* Simulate redstone *waaaaaaaay* faster than Minecraft.
* Parse redstone contraptions from [schematic files](https://minecraft.fandom.com/wiki/Schematic_file_format).
* Provide automated testing of redstone contraptions.
* Provide useful visualizations through graphs ([DOT](https://en.wikipedia.org/wiki/DOT_(graph_description_language)))
  and traces ([VCD](https://en.wikipedia.org/wiki/Value_change_dump)).
* Perform complex analysis such as finding the worst-case path (nyi)

We also want to clearly indicate what was not in the scope of this project:

* Simulating redstone *exactly* as Minecraft does it. There are some edge-cases in which redstone is really wacky, we
  will mention these edge-cases later when we encounter them.
* Simulating moving parts and most tile-entities. We were only interested in *vanilla* solid-state blocks (sorry piston
  lovers!).

<br />

# How does Redstone work anyway?

For those not familiar with Minecraft redstone, or have maybe forgotten by now, this section will serve as a brief
overview of all the blocks we will be simulating.

## Weak and Strong Power

In Minecraft, blocks emit a form of power we will call *signal strength*, which can be categorized into two variations:
*weak power* and *strong power*. It is important to note that different blocks emit different types of power, and that
not all blocks can accept both types of power.

In both variations, the signal strength ranges from 0 to 15. If a block receives a signal strength of at least 1, it is
considered to be in the *on* state. Otherwise, it is considered to be in the *off* state.

In the following table we show how blocks can connect. An "X" means the blocks can connect unconditionally, whereas some
blocks can only connect to the rear of other blocks.

| Emitter \ Receiver | Wire | Solid (Weak) | Solid (Strong) | Repeater | Torch | Comparator |
|--------------------|------|--------------|----------------|----------|-------|------------|
| Wire               | X    | X            |                | Rear     |       | X          |
| Solid (Weak)       |      |              |                | Rear     | X     | Rear       |
| Solid (Strong)     | X    |              |                | Rear     | X     | Rear       |
| Redstone Block     | X    |              |                | Rear     | X     | X          |
| Repeater           | X    |              | X              | X        |       | X          |
| Torch              | X    |              | *1             | Rear     |       | Rear       |
| Comparator         | X    |              | X              | X        |       | X          |

1. Only if the solid block is *above* the torch.

## Redstone Wire

Redstone wire serves as the most basic building block for any redstone contraption. It can connect to other blocks in
any cardinal direction and can traverse one level up or down.

When redstone wire is placed on a block, it provides weak power to that block as well as to all the blocks it faces.
However, when wires connect to other wires, they lose 1 signal strength per block traveled.
This means that after traversing 15 blocks, the signal strength of the wire will be completely depleted.

{{ image(src="assets/redstone_wire.png", alt="Redstone wire *on* and *off*.", position="center", style="width: 60%;
border-radius: 8px;") }}

## Solid Blocks

Redstone signals can travel through regular solid blocks. It is important to note that solid blocks behave differently
when they receive weak and strong powered signals. A solid block will simply act as a signal extender for strong powered
signals. However, for weak powered signals, it will clamp the signal strength between 0 and 1. This distinction lead us
to split these blocks into two distinct *construction blocks* (this will be discussed later).

## Redstone Blocks

Redstone blocks are the simplest blocks to understand. They provide weak power to any block they touch (excluding solid
blocks) with a constant signal strength of 15.

## Repeaters

- Boost signal strength
- Delay 1-4
- Extending signal (only on-s)
- Locking

## Torches

- Invert signal
- Delay 1
- Burnout?

## Comparators

Two modes:

- Compare: Output strength = rear if side <= rear
- Subtract: Output strength = rear - side

## Debouncing

Repeaters, torches, and comparators in minecraft have a behaviour that is not well documented that we like to call
*debouncing*. These blocks can't see 1-tick redstone input pulses, they will completely ignore these, but only
*sometimes*. We're still not entirely sure how exactly this works, but we decided that no sane person would rely on this
behaviour in their redstone circuits, so we decided against including this behaviour in the simulator. If you do know
exactly how this works, please let us know!

<br />

# Simulating the world

## Direct simulation

- World is stored as a `Vec<Vec<Vec<Vec<Block>>>>` (4D!)
- The first 3 vecs are for x/y/z of the blocks
- The last vec is for if a position contains no or multiple blocks (will explain later)

- We keep track of a list of blocks whose inputs were updated
- These need to be updated, they calculate their output power from input power
- Some blocks need a *late update*, after all other blocks have been processed they will be processed again
    - For example: Torches only change output then, effectively changing their output the next tick

## Blocks

- Air is empty vec
- Solid creates two blocks for weak/solid (expand on this, show example)
- All other blocks just one block

## Simulating the world as a graph

## Creating a graph

- Observation!: Each block has 3 types of neighbours
    - Rear inputs
    - Side inputs (not relevant for some blocks)
    - Output
- These lists are static, they don't change during simulation (this is the advantage of not simulating pistons)
- Why not pre-compute them and store the blocks as nodes in a graph
- Picture: Side-by-side redstone circuit and corresponding graph

To know if there is an edge between two blocks in the graph, check between each neighbouring pair of blocks:

- Can output: Does this block output power in the direction of this neighbour?
    - For example: Repeater can only output power in one direction
- Can input: Does this block input power in the direction of this neighbour? (If so, rear/side?)
    - For example: Repeater has rear input in one direction, side inputs in two directions
- Can connect: Are these two blocks "compatible", see table:

| Emitter \ Receiver | Wire | Solid (Weak) | Solid (Strong) | Repeater | Torch | Comparator |
|--------------------|------|--------------|----------------|----------|-------|------------|
| Wire               | X    | X            |                | Rear     |       | X          |
| Solid (Weak)       |      |              |                | Rear     | X     | Rear       |
| Solid (Strong)     | X    |              |                | Rear     | X     | Rear       |
| Redstone Block     | X    |              |                | Rear     | X     | X          |
| Repeater           | X    |              | X              | X        |       | X          |
| Torch              | X    |              | *1             | Rear     |       | Rear       |
| Comparator         | X    |              | X              | X        |       | X          |

1. Only if the solid block is *above* the torch.

We can now simulate blocks in the graph as we did before, during direct simulation.

## Pruning the graph

<br />

# Testing, lots of testing

[//]: # (This chapter will contain information on how we did testing, in code and Minecraft itself to discover how it works.)

[//]: # (picture of our tests in the world)

<br />

# Visualizations

<br />

# Other stuff

<br />

# Coding experience
