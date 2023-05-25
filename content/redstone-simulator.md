+++
title = "Building a Redstone Simulator in Rust"
date = 2023-05-25

[taxonomies]
tags = ["rust", "minecraft", "redstone"]
+++

# Why we do what we do
Recently we decided to play some Minecraft again, although maybe not in the vanilla survival way you might know very well. 
Instead, we wanted to put Minecraft to the test by building a fully fledged 8-bit CPU using the Redstone blocks that the game offers.
After a while we had the basis for our CPU down, but before implementing the rest of the functionality, 
we were interested in the worst-case response times. For those interested, the version of the CPU as of writing this blog can be seen in 
image (refer to image). Obviously this contraption has become too unwieldy to quickly determine where the worst-case delays would manifest.
This led us to come up with the idea of using software outside of Minecraft to simulate our CPU, allowing us to more easily
probe signals and analyze the behaviour. This is where the idea of our Redstone Simulator originated.

The source code for this project is available on [GitHub](https://github.com/JonathanBrouwer/redstone-simulator) and was written for Minecraft version 1.19. 

{{ image(src="assets/cpu.png", alt="CPU in Minecraft", position="left", style="border-radius: 8px;") }}

# A redstone simulator
So with the idea of a Redstone Simulator in mind, we wanted to clearly define the scope of this project. 
Most importantly we set the following goals for our project:
* Allow fine-grained probing of redstone contraptions.
* Simulate redstone *waaaaaaaay* faster than Minecraft.
* Parse redstone contraptions from [schematic files](https://minecraft.fandom.com/wiki/Schematic_file_format).
* Provide automated testing of redstone contraptions.
* Provide useful visualizations through graphs ([DOT](https://en.wikipedia.org/wiki/DOT_(graph_description_language))) and traces ([VCD](https://en.wikipedia.org/wiki/Value_change_dump)).
* Perform complex analysis such as finding the worst-case path (nyi)

We also want to clearly indicate what was not in the scope of this project:
* Simulating redstone *exactly* as Minecraft does it. There are some edge-cases in which redstone is really wacky, we will mention these edge-cases later when we encounter them.
* Simulating moving parts and most tile-entities. We were only interested in *vanilla* solid-state blocks (sorry piston lovers!).

# How does Redstone work anyways?
For those not familiar with Minecraft redstone, or have maybe forgotten by now, this section will serve as a brief overview of all the blocks we will be simulating.

## Redstone Wire
Redstone wire forms the most basic building block for any redstone contraption. 
It carries what we will refer to as *signal strength* which can range anywhere from 0 up to and including 15.
Any block receiving a signal from a redstone wire will be considered *on* if the signal strength is above 0, otherwise it will be *off*.
Redstone wire can connect in any cardinal direction and also go up and down one level. 
It powers the block it sits upon and all the blocks that it faces. Wires connecting to other wires will lose 1 signal strength per block traveled. 
This means that after 15 blocks it will have lost all of its strength.

{{ image(src="assets/redstone_wire.png", alt="Redstone wire *on* and *off*.", position="center", style="width: 60%; border-radius: 8px;") }}
## Redstone Blocks
Redstone blocks are by far the easiest block to understand. It simply powers any block it touches (except for solid blocks) with a signal strength of 15.

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

Repeaters, torches, and comparators in minecraft have a behaviour that is not well documented that we like to call "debouncing".
These blocks can't see 1-tick redstone input pulses, they will completely ignore these, but only *sometimes*. 
We're still not entirely sure how exactly this works, but we decided that no sane person would rely on this behaviour in their redstone circuits, so we decided against including this behaviour in the simulator.
If you do know exactly how this works, please let us know!

# Testing, lots of testing

[//]: # (This chapter will contain information on how we did testing, in code and Minecraft itself to discover how it works.)
[//]: # (picture of our tests in the world)


# Visualizations

# Other stuff

# Coding experience

# Simulating the world directly

# Simulating the world as a graph

# Pruning the graph
