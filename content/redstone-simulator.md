+++
title = "Building a Redstone Simulator in Rust"
date = 2023-05-25

[taxonomies]
tags = ["rust", "minecraft", "redstone"]
+++

[//]: # (TODO:)

[//]: # (- clarify the goal in "Parsing the World from a Schematic")
[//]: # (- Why rust?)
[//]: # (- performance ✨)
[//]: # (- fix tags)

# Table of Contents

<!-- TOC -->
* [Table of Contents](#table-of-contents)
* [Why We Do What We Do](#why-we-do-what-we-do)
* [A Redstone Simulator](#a-redstone-simulator)
* [How Does Redstone Work Anyway?](#how-does-redstone-work-anyway)
  * [Redstone Timing and Updates](#redstone-timing-and-updates)
  * [Weak and Strong Power](#weak-and-strong-power)
  * [Redstone Wire](#redstone-wire)
  * [Solid Blocks](#solid-blocks)
  * [Redstone Blocks](#redstone-blocks)
  * [Repeaters](#repeaters)
  * [Torches](#torches)
  * [Comparators](#comparators)
  * [Triggers and Probes](#triggers-and-probes)
  * [Debouncing](#debouncing)
* [Parsing the World from a Schematic](#parsing-the-world-from-a-schematic)
* [Simulating the World](#simulating-the-world)
  * [Array-Based Simulation](#array-based-simulation)
    * [Intra-Tick Updates](#intra-tick-updates)
    * [Late-Updates](#late-updates)
  * [Graph-Based Simulation](#graph-based-simulation)
  * [Creating a Graph](#creating-a-graph)
  * [Pruning the Graph](#pruning-the-graph)
    * [1. Prune Dead Nodes](#1-prune-dead-nodes)
    * [2. Prune Redstone](#2-prune-redstone)
    * [3. Prune Duplicate Edges](#3-prune-duplicate-edges)
    * [4. Prune Groups](#4-prune-groups)
    * [5. Prune Untraversable Edges](#5-prune-untraversable-edges)
    * [6. Prune Irrelevant Nodes](#6-prune-irrelevant-nodes)
    * [7. Prune Subtractor Edges](#7-prune-subtractor-edges)
    * [8. Prune Constants](#8-prune-constants)
    * [9. Replace Simple Repeaters](#9-replace-simple-repeaters)
* [Testing, Lots of Testing](#testing-lots-of-testing)
* [Visualizations](#visualizations)
* [The "Minecraft is Cursed" Section](#the-minecraft-is-cursed-section)
  * [Order Matters for Tick Updates](#order-matters-for-tick-updates)
  * [Comparators are Tile Entities](#comparators-are-tile-entities)
* [Coding Experience](#coding-experience)
<!-- TOC -->

<br />

# Why We Do What We Do

Recently, we decided to delve back into playing some Minecraft, but maybe not in the way you might be accustomed to.
Instead of playing the familiar vanilla survival mode, we set out to put Minecraft (and ourselves) to the test by
building a fully fledged 8-bit CPU using the game's redstone blocks. If you're curious, you can see the most recent
version of our CPU in the image below.

As we made progress and laid the foundation for our CPU, we realized the need to determine the worst-case response times
before proceeding with additional functionality. However, due to the growing complexity of our contraption, it became
increasingly difficult to identify the specific areas where delays were most likely to occur. To address this issue, we
came up with the idea of using external software to simulate our CPU. This approach would enable us to analyze signal
behavior and probe for potential bottlenecks more easily. And so, the idea for our <emph>Redstone Simulator</emph> was
born.

In this blog, we will delve into the development and functionality of our Redstone Simulator, exploring the difficulties
and curiosities we came across along the way. Stay tuned for an exciting journey into the world of Minecraft's digital
circuitry!

The source code for this project is available on [GitHub](https://github.com/JonathanBrouwer/redstone-simulator) and was
written for Minecraft version 1.19.

{{ image(src="assets/redstone-simulator/cpu.jpeg", alt="CPU in Minecraft", position="left") }}

<br />

# A Redstone Simulator

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

# How Does Redstone Work Anyway?

For those not familiar with Minecraft redstone, or have maybe forgotten by now, this section will serve as a brief
overview of all the blocks we will be simulating. The following enum shows all the possible blocks that can be
simulated.

```rust
pub enum CBlock {
    Redstone(CRedstone),
    SolidWeak(CSolidWeak),
    SolidStrong(CSolidStrong),
    Trigger(CTrigger),
    Probe(CProbe),
    Repeater(CRepeater),
    SRepeater(CSRepeater),
    RedstoneBlock(CRedstoneBlock),
    Torch(CTorch),
    Comparator(CComparator),
}
```

## Redstone Timing and Updates

Before we dive into how the different blocks work, it seems prudent to first explain how Minecraft actually updates
them.
In simple terms, Minecraft performs a redstone update once every 0.1 seconds.
These updates are commonly referred to as redstone ticks - which we will shorten to ticks for the remainder of this
blog.
One way we will talk about the performance of our simulation later is then by mentioning the <emph>TPS</emph> (ticks per
second).

So what happens during an update?
Minecraft keeps track of a list of blocks that need to be updated at the start of a tick, and then proceeds to update
them one-by-one.
Updating blocks might actually cause *more* updates to occur in the same tick, and some updates might be scheduled for
the next tick.
Important to note is that the order in which the blocks are updated from this list can influence the behaviour of the
redstone contraption.
For this reason we have decided *not* to go with this implementation for our redstone simulations.

## Weak and Strong Power

In Minecraft, blocks emit a form of power we will call <emph>signal strength</emph>, which can be categorized into two
variations:
<emph>weak power</emph> and <emph>strong power</emph>. It is important to note that different blocks emit different
types of power, and that
not all blocks can accept both types of power.

In both variations, the signal strength ranges from 0 to 15. If a block receives a signal strength of at least 1, it is
considered to be in the <emph>on</emph> state. Otherwise, it is considered to be in the <emph>off</emph> state.

In the following table we show how blocks can connect. An "X" means the blocks can connect unconditionally, whereas some
blocks can only connect to the rear of other blocks.

<table style="text-align: center">
  <tr>
    <th>Emitter \ Receiver</th>
    <th>Wire</th>
    <th>Solid (Weak)</th>
    <th>Solid (Strong)</th>
    <th>Repeater</th>
    <th>Torch</th>
    <th>Comparator</th>
  </tr>
  <tr>
    <th>Wire</th>
    <td>X</td>
    <td>X</td>
    <td></td>
    <td>Rear</td>
    <td></td>
    <td>X</td>
  </tr>
  <tr>
    <th>Solid (Weak)</th>
    <td></td>
    <td></td>
    <td></td>
    <td>Rear</td>
    <td>X</td>
    <td>Rear</td>
  </tr>
  <tr>
    <th>Solid (Strong)</th>
    <td>X</td>
    <td></td>
    <td></td>
    <td>Rear</td>
    <td>X</td>
    <td>Rear</td>
  </tr>
  <tr>
    <th>Redstone Block</th>
    <td>X</td>
    <td></td>
    <td></td>
    <td>Rear</td>
    <td>X</td>
    <td>X</td>
  </tr>
  <tr>
    <th>Repeater</th>
    <td>X</td>
    <td></td>
    <td>X</td>
    <td>X</td>
    <td></td>
    <td>X</td>
  </tr>
  <tr>
    <th>Torch</th>
    <td>X</td>
    <td></td>
    <td>*1</td>
    <td>Rear</td>
    <td></td>
    <td>Rear</td>
  </tr>
  <tr>
    <th>Comparator</th>
    <td>X</td>
    <td></td>
    <td>X</td>
    <td>X</td>
    <td></td>
    <td>X</td>
  </tr>
</table>

1. Only if the solid block is *above* the torch.

## Redstone Wire

Redstone wire serves as the most basic building block for any redstone contraption. It can connect to other blocks in
any cardinal direction and can traverse one level up or down.

When redstone wire is placed on a block, it provides weak power to that block as well as to all the blocks it faces.
However, when wires connect to other wires, they lose 1 signal strength per block traveled.
This means that after traversing 15 blocks, the signal strength of the wire will be completely depleted.
This can be seen in the image below on the left.
The ways in which redstone can connect up and down is shown in image on the right.
Note that transparent blocks such as glass allow connections upwards *but* not downwards.
Also note that signals moving upwards are not blocked by transparent blocks.

{{ image(src="assets/redstone-simulator/redstone_wire1.jpeg", position="inline", style="width: 49%;") }}
{{ image(src="assets/redstone-simulator/redstone_wire2.jpeg", position="inline", style="width: 49%;") }}

## Solid Blocks

Redstone signals can travel through regular solid blocks. It is important to note that solid blocks behave differently
when they receive and emit weak and strong powered signals (see [table](#weak-and-strong-power)).
For this reason we have split the solid block into two different blocks: Solid (Weak) and Solid (Strong), that just
happen to occupy the same space.
This distinction will also lead to an easier implementation later on when we start talking about <emph>construction
blocks</emph>.
weird
In the image below the different behaviours of solid blocks are presented.
From left to right we see a block that receives weak power, a block that receives strong power and lastly a block that
receives weak power.
Note that in the left case the signal stays on because a repeater is used.

{{ image(src="assets/redstone-simulator/solids.jpeg", position="center", style="width: 80%;") }}

## Redstone Blocks

Redstone blocks are the simplest blocks to understand. They provide weak power to any block they touch (excluding solid
blocks) with a signal strength of 15.

In the image below, the behaviour of a redstone block is shown. It simply acts as a constant power source to all its
surrounding blocks.

{{ image(src="assets/redstone-simulator/redstone_block.jpeg", position="center", style="width: 80%;") }}

## Repeaters

Repeaters are directed blocks that, as the name implies, repeat signals from their input side (rear) to their output
side (front).
This means that for any signal that is *on*, the repeater will emit a strong power of 15.

Additionally, repeaters have a configurable delay setting that ranges from 1 tick to 4 ticks.
It is common to refer to a repeater by the number of ticks delay it causes, e.g. *a 2-tick repeater*.
This means that it will take however many ticks of delay was set before a change on the input side is detected on the
output side. The gif on the left below shows how a clock can be made using different delays.
There are some technicalities involved with how repeater timing works precisely, which are discussed in the subsection
on [Debouncing](#debouncing).

Finally, repeaters can be <emph>locked</emph> by providing an *on* signal from either of their two side inputs.
A locked repeater will retain the signal that was emitted on the tick before being locked.
An example of a locking repeater can be seen in the image below on the right.

<video width="49%" autoplay muted loop style="border-radius: 8px">
  <source src="/assets/redstone-simulator/repeater_loop.mp4" type="video/mp4">
</video> 
{{ image(src="assets/redstone-simulator/repeater_locking.jpeg", position="inline", style="width: 49%;") }}

## Torches

Torches are also directed blocks. They invert the signal they receive on their input side, and output it to all other
sides.
This means that if the input signal is *on*, the torch will output a strong power of 0, while if the input signal is
*off*, it will output a strong power of 15.
Additionally, torches, like repeaters, introduce a delay of 1 tick before propagating signals.

It should be noted that torches <emph>burn out</emph> if they are toggled more than 8 times in 30 ticks.
We have decided *not* to include this behaviour since it is unlikely to be present in most redstone contraptions.

{{ image(src="assets/redstone-simulator/torch.jpeg", position="center", style="width: 80%;") }}

## Comparators

Redstone comparators are blocks that can be in two modes: <emph>Compare</emph> and <emph>Subtract</emph>.
It compares the input signal on the rear to the input signal on either of its sides and then, depending on the mode,
will output a signal according to the following:

```rust
let output = match mode {
Mode::Compare => if side < = rear { rear } else { 0 },
Mode::Subtract => max(rear - side, 0),
}
```

This behaviour can be seen in the image below on the left.
Where the left comparator is in compare mode, and the right comparator is in subtract mode - indicated by the front
torch being lit.

Additionally, comparators can read tile entities behind them and see that as their rear input.
The specifics of how the input signal should be calculated can be
found [here](https://minecraft.fandom.com/wiki/Redstone_Comparator#Measure_block_state).
We have *not* fully implemented this behaviour.
Instead, we opted to only check for *furnace* blocks behind comparators and always set the input signal strength as 1.
An example of how this looks in Minecraft can be seen in the image below on the right.

{{ image(src="assets/redstone-simulator/comp1.jpeg", position="inline", style="width: 49%;") }}
{{ image(src="assets/redstone-simulator/comp2.jpeg", position="inline", style="width: 49%;") }}

## Triggers and Probes

Triggers and probes are not really blocks that exist in Minecraft, but that will be necessary for any useful analysis.
During the parsing of the world, we read in all gold blocks and lightning rods as triggers and all diamond blocks as
probes.
Additionally, probes can be given by a name by placing a sign on any of its faces.
The name will be the text on the first line of the sign, otherwise the probe simply gets its coordinates as its name.

Functionally speaking triggers and probes will both act as solid blocks, where the output power of a redstone block can
be controlled and the input power of a probe can be measured.
This functionality can be used to write tests (which we will speak more about [later](#testing-lots-of-testing)). An
example of this can be seen below.

{{ image(src="assets/redstone-simulator/probes_and_triggers.jpeg", position="center", style="width: 100%;") }}

## Debouncing

Repeaters, torches, and comparators in minecraft have a behaviour that is not well documented that we like to call
<emph>debouncing</emph>. These blocks can't see 1-tick redstone input pulses, they will completely ignore these, but
only
*sometimes*. We're still not entirely sure how exactly this works, but we decided that no sane person would rely on this
behaviour in their redstone circuits, so we decided against including this behaviour in the simulator. If you do know
exactly how this works, please let us know!

<br />

# Parsing the World from a Schematic

Now that we know how all the blocks are supposed to work, we can move on to parsing the world.
Our objective will be to construct a 4D array of blocks, where the first 3 axes represent positions and the final axis
represents the blocks occupying those positions.

As mentioned before, redstone contraptions are commonly saved
as [schematics](https://minecraft.fandom.com/wiki/Schematic_file_format).
These schematics are stored in [Named Binary Tag](https://minecraft.fandom.com/wiki/NBT_format) (NBT) format.
We would highly recommend playing around with [NBTExplorer](https://github.com/jaquadro/NBTExplorer) if you want to see
what your schematics look like (it definitely helped us!).

In order to make these files more manageable we constructed the following structs using
the [hematite-nbt](https://crates.io/crates/hematite-nbt) and [serde](https://crates.io/crates/serde) crates:

```rust
pub struct SchemFormat {
    // length across x-axis
    pub width: i16,
    // length across y-axis
    pub height: i16,
    // length across z-axis
    pub length: i16,

    pub block_data: Vec<i8>,
    pub palette: HashMap<String, i32>,

    pub block_entities: Vec<SchemBlockEntity>,

    // ...
}

pub struct SchemBlockEntity {
    pub id: String,
    pub pos: Vec<i32>,
    pub props: HashMap<String, Value>,
}
```

The goal now is to map the `block_data` field, using the `palette` field, to a usable data structure for holding the
world data.
Unfortunately there are two issues that arise, the first is that `palette` stores a map from string identifiers to
unique `i32`s.
Instead, we would like to have a vector to look up what construction blocks should be where.
The second issue is that for we need to decode the `block_data` field, which contains indices into the palette.
Let's tackle the first issue first!

We will create a 2D vector of construction blocks, where the final axis can contain multiple construction blocks, such
as for solid blocks.
This can be achieved by sorting the given palette by its keys and then mapping these keys to construction blocks.

```rust
fn create_palette(format: &SchemFormat) -> Vec<Vec<CBlock>> {
    format
        .palette
        .iter()
        .sorted_by_key(|(_, i)| *i)
        .map(|(id, _)| CBlock::from_id(id.as_str()))
        .collect()
}
```

The second issue is a bit more involved.
The `block_data` field stores the blocks in the schematic in YZX order, with the X coordinate varying the fastest.
However, if you paid close attention you might have noticed that it contains `i8`s instead of `i32`s, indicating that
there is something extra happening.

What is happening is that the block data is encoded using
a [variable-width encoding](https://en.wikipedia.org/wiki/Variable-width_encoding),
which means that if the leading bit of a byte is 1, the next byte is also a part of the block's index. Here are some
examples:

* \[0b0xxx_xxxx\] -> 0b0xxx_xxxx.
* \[0b1yyy_yyyy, 0b0xxx_xxxx\] -> 0b00xx_xxxx_xyyy_yyyy.

To read the next identifier, we used the following code:

```rust
let mut read_head = 0;
let mut read_next = | | {
let mut ix: usize = 0;
for j in 0.. {
let next = format.block_data[read_head];
ix |= (next as usize & 0b0111_1111) << (j * 7);
read_head += 1;

if next > = 0 {
break;
}
}
ix
};
```

A block entity (also known as tile entity) is extra data that is associated with a block.
This is used because the limited set of data that can be stored in a blocks properties is often not enough.
For example, chests can store many items. [Comparators are also block entities](#comparators-are-tile-entities).

<br />

# Simulating the World

Now that we know how all the blocks we will be simulating work, we can start thinking about the logic involved in
simulating a world.

## Array-Based Simulation

Our first attempt at simulating redstone was an <emph>array-based</emph> simulation.
We simulated the world the same way that Minecraft does, by storing the blocks in a 3D list that represents the world.
However, because air blocks generate no blocks, and solid blocks generate two blocks (weak and strong), we actually need
a 4D list, the last axis representing the list of blocks at a certain position,

Each tick consists of two phases: intra-tick updates and end-of-tick updates.
At the start of simulating a tick we start by executing all intra-tick updates before executing the end-of-tick updates.
We will now explain these two updates in more detail:

### Intra-Tick Updates

During the intra-tick updates phase we keep track of a queue of blocks that still has to be updated.

For each block that was queued to be updated calculates its output from the output of its neighbors.
If the block's output has changed during this calculation, it adds its neighbours to the queue of blocks to be updated
during this tick.
This ensures that the neighboring blocks are updated accordingly.
This process continues until the update queue is empty.

Some blocks such as repeaters and torches change their state not in this tick, but after it.
When the input of these blocks changes the blocks are queued for a late-update, which happens at the end of the current
tick.

### Late-Updates

In the late-updates phase, we go through the list of blocks that was marked for a late-update in the previous phase.
We check if the blocks output should change, and if it should we add its neighbours to the intra-tick updates queue of
the next tick.

## Graph-Based Simulation

The array-based representation of the world is the one that is closest to reality, and it would allow simulating blocks
such as pistons more easily.
It is however also very inefficient since a lot of an updated block's neighbours are put into the update lists that
might not actually need to be updated.

An observation that we made is that each block has three lists of neighbours, and these lists remain constant throughout
the simulation.
The lists are as follows:

* Output neighbors: Blocks that are powered by the current block.
* Rear inputs: Blocks that provide power to the current block from the rear. For blocks with no side inputs, every input
  is considered as a rear input.
* Side inputs: Blocks that provide power to the current block from the side.

We can represent this data as a <emph>graph</emph> where the blocks are nodes, and the connections between them are
edges.
These edges are divided into rear and side edges, which are weighted by the signal strength loss between the source and
target block.
Below we show the same circuit both in minecraft and as a graph.

{{ image(src="assets/redstone-simulator/example_minecraft.jpeg", position="inline", style="width: 49%;") }}
{{ image(src="assets/redstone-simulator/example_graph.png", position="inline", style="width: 49%;") }}

Notice that:

* The color of the graph node corresponds to the type of block
* The edge between the repeaters is gray: It is a side edge.
* The solid block generated a weak and strong node. The strong node is unused here, which can be seen by the fact that
  it only has outgoing neighbours.

## Creating a Graph

So now the question becomes, how do we turn our world data into a usable graph?
Adding the nodes to the graph is simple enough. However, determining what edges should be added is more involved.
One problem is that there are many possible connections between the different blocks, where the facing of both blocks
matters.
Additionally, we need to differentiate between *side* and *rear* edges.

Our solution to tackling this problem was to split up the logic of whether an edge should exist between two blocks into
three parts.
For each block and its neighbours we check the following:

1. Does the source block output power in the direction of its neighbour?
2. Does the neighbor receive power from the direction of the source block?
3. Can the source block and its neighbour connect? See [table](#weak-and-strong-power).

If the answer to all these questions is "yes", then an edge will be created between the source block and its neighbour.
In order to determine whether it should be a *side* or *rear* edge we return some extra information while answering
question the second question.

We can now simulate blocks in the graph as we did before. However, now we only need to update neighbours that can
actually be updated! Yay!

## Pruning the Graph

But why stop there? With our world now stored as a graph we have unlocked the superpower of <emph>pruning</emph>!
You might have noticed in the previous section that there was a strong solid block that had no incoming neighbours,
meaning that it is unnecessary to simulate.
By removing such nodes and the edges connected to them, we can drastically reduce the complexity and size of the graph.

We actually managed to come up with 9 different pruning methods, which when combined, managed to reduce the number of
nodes and edges by a factor of 75!
In the subsections below we discuss the following methods:

1. [Prune dead nodes](#1-prune-dead-nodes).
2. [Prune redstone](#2-prune-redstone).
3. [Prune duplicate edges](#3-prune-duplicate-edges).
4. [Prune groups](#4-prune-groups).
5. [Prune untraversable edges](#5-prune-untraversable-edges).
6. [Prune irrelevant nodes](#6-prune-irrelevant-nodes).
7. [Prune subtractor edges](#7-prune-subtractor-edges).
8. [Prune constants](#8-prune-constants).
9. [Replace simple repeaters](#9-replace-simple-repeaters).

### 1. Prune Dead Nodes

The simplest pruning method we came up with was to remove <emph>dead nodes</emph> such as in the example given above.
To qualify as a dead node, a node must not be a trigger or probe, and it must have no incoming or outgoing edges, or both.*

Note that the removal of dead nodes can lead to the creation of new dead nodes.
In fancy terms this operation is not *idempotent*.
To make this pruning method idempotent, we can iteratively remove dead nodes until no more dead nodes remain in the graph.

\* An exception is that we cannot remove redstone blocks or redstone torches with no incoming edges since they still provide a constant output signal.
We consider this special case when we [prune constants](#8-prune-constants).

{{ image(src="assets/redstone-simulator/prune1.png", alt="A graph showing an example of prune 1", position="center", style="background: white; width: 100%;") }}

### 2. Prune Redstone

The next pruning method attempts to remove all the redstone wires from the computer.

Wait <emph>WHAT</emph>?

Yes, you read that right! 
The graph structure actually allows us to completely abstract all wires away by using edge weights.
In fact, it even allows us to do the same for solid blocks!

The core idea is to make the weights of the edges represent the type of connection (side or rear) that exists between blocks and a number representing the signal strength loss between them.

The following two points justify why we are allowed to do this:
* Redstone wires and solid blocks only have intra-tick updates, making their behaviour seem *instant*. So, we only need edges between blocks with delays.
* Blocks that are connected through redstone wires and solid blocks will always take the same *shortest* path. So, we can collapse everything to this path.

{{ image(src="assets/redstone-simulator/prune2.png", alt="A graph showing an example of prune 2", position="center", style="background: white; width: 100%;") }}


### 3. Prune Duplicate Edges

Another method involves removing <emph>duplicate edges</emph> between blocks.
Nodes that have multiple edges of the same type only need to keep the edge with the lowest weight.
This is allowed because edges with higher weights will always be overshadowed by edges with lower weights.
In simple terms, redstone power always follows the *path of least resistance*.

{{ image(src="assets/redstone-simulator/prune3.png", alt="A graph showing an example of prune 3.", position="center", style="background: white; width: 100%;") }}


### 4. Prune Groups

This method attempts to condense groups of nodes into a single node.
We consider a set of torches or repeaters a <emph>group</emph> when they are all driven by a torch or repeater.

Such a group can then be replaced by a single torch or repeater.
The single node will have an incoming edge connected to the driving torch or repeater and all outgoing edges from the blocks in the group.

This substitution is allowed because the entire group is always driven to be *on* or *off*.
For this reason, the condensed node maintains the same behaviour when compared to the original connections.

{{ image(src="assets/redstone-simulator/prune4.png", alt="A graph showing an example of prune 4", position="center", style="background: white; width: 100%;") }}

### 5. Prune Untraversable Edges

A very simple and effective prune is to remove all edges with weights of at least 15.
Since redstone has a maximum signal strength of 15, these edges will never provide an *on* signal.
These <emph>untraversable edges</emph> can therefore be safely removed.



### 6. Prune Irrelevant Nodes

Remember that we placed probes in our redstone circuit to measure its behaviour. 
Some parts of the circuit may not have any effect on these probes. These parts can be safely removed.

{{ image(src="assets/redstone-simulator/prune6.png", alt="A graph showing an example of prune 6", position="center", style="background: white; width: 100%;") }}

### 7. Prune Subtractor Edges

If a block 
- is connected to both the rear and side of a comparator
- that comparator is in subtract mode
- the rear edge has a higher signal strength loss than the side edge
Then
- the rear edge can be removed
- if the rear and side are both on, `rear - side` will always be 0.
- the side cannot be removed because another rear edge might provide a stronger signal

### 8. Prune Constants

An edge case that we missed during dead node pruning is that some nodes may provide a constant 

### 9. Replace Simple Repeaters

technically not a prune since we do not remove any edges or nodes

<br />

# Testing, Lots of Testing

Testing was an integral part to making this project a success.
Whenever we could, we tried to make tests on the behaviour we expected of some to be implemented features.
This not only helped us easily identify regressions in our code, but it also deepened our understanding of the
interactions between different blocks.

We created both unit tests and integration tests.
The unit tests can be seen in the image below.
As you can see we kind of went crazy, but believe it or not, this actually made isolating bugs a lot easier.

For the integration tests, we tested an 8-bit adder and our computer loaded with a Fibonacci program.
In total, we have written a staggering 127 combined unit and integration tests!

{{ image(src="assets/redstone-simulator/testing.jpeg", position="center", style="width: 100%;") }}

In order to write tests, we first made schematic files of the circuits to be tested.
Next, we made assertions on the expected behaviour of this circuit.
To make this process manageable we decided to make use of the following <emph>rust macro</emph>:

```rs
macro_rules! test {
    ($file:literal, $name:ident, $triggers: expr; $($b:expr),*) => {
        #[test]
        fn $name() {
            use std::fs::File;
            use redstone_simulator::world::World;

            let f = File::open(format!("./schematics/{}.schem", $file)).unwrap();
            let mut world = World::from(f);

            let mut triggers = $triggers;

            $(
            assert_eq!(world.get_probe(stringify!($name)).unwrap(),$b);
            if triggers > 0 {
                world.step_with_trigger();
                triggers -= 1;
            } else {
                world.step();
            }
            )*
        }
    };
}
```

This macro takes the following arguments:

* file: The name of the schematic file to use for the test.
* name: The name of the probe (which will also be the test name).
* triggers: The number of ticks that the triggers will be powered for at the start of the test.
* booleans: A list of booleans indicating whether the probe is expected to be *on* or *off* on each tick.

A typical test could then be constructed like this:

```rust
test!("<file>", <name>, <triggers>; F, F, T, F);
```

<br />

# Visualizations

{{ image(src="assets/redstone-simulator/cpu_graph.svg", alt="Graph of redstone computer visualized with Gephi.",
position="center", style="width: 100%; margin: 10px;") }}

{{ image(src="assets/redstone-simulator/cpu_trace.png", alt="Trace of simulated redstone computer in GTKWave.",
position="center", style="width: 100%;") }}

<br />

# The "Minecraft is Cursed" Section

## Order Matters for Tick Updates

The order in which blocks are processed depends on where the update starts.
The state at the end of the tick can be different depending on this order.
In the image below you can see how two levers at different locations can produce different outcomes.

<video width="100%" autoplay muted loop style="border-radius: 8px">
  <source src="/assets/redstone-simulator/weird1.mp4" type="video/mp4">
</video>

## Comparators are Tile Entities

All redstone blocks store their output signal strength as part of their block properties.
This makes sense since it is just a single number from 0 to 15, which could easily just be a property.

However, for legacy reasons, comparators are a tile entity.
[Pre-flattening](https://minecraft.fandom.com/wiki/Java_Edition_data_values/Pre-flattening), Minecraft used to have only
2 bytes to store the entire state of a block, one for a block id and one for additional data.
Comparators need to store whether they are in compare or subtract mode (1 bit) and they need to store their signal
strength (4 bits).
This could be packed into one byte of additional data, but the developers decided against this for some reason.

<br />

# Coding Experience
