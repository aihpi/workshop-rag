# LONG SHORT-TERM MEMORY

NEURAL COMPUTATION 9(8):1735–1780, 1997

**Sepp Hochreiter** Fakultät für Informatik Technische Universität München 80290 München, Germany hochreit@informatik.tu-muenchen.de http://www7.informatik.tu-muenchen.de/~hochreit

**Jürgen Schmidhuber** IDSIA Corso Elvezia 36 6900 Lugano, Switzerland juergen@idsia.ch http://www.idsia.ch/~juergen

## Abstract

Learning to store information over extended time intervals via recurrent backpropagation takes a very long time, mostly due to insufficient, decaying error back flow. We briefly review Hochreiter’s 1991 analysis of this problem, then address it by introducing a novel, efficient, gradient-based method called “Long Short-Term Memory” (LSTM). Truncating the gradient where this does not do harm, LSTM can learn to bridge minimal time lags in excess of 1000 discrete time steps by enforcing *constant* error flow through “constant error carrousels” within special units. Multiplicative gate units learn to open and close access to the constant error flow. LSTM is local in space and time; its computational complexity per time step and weight is $O(1)$. Our experiments with artificial data involve local, distributed, real-valued, and noisy pattern representations. In comparisons with RTRL, BPTT, Recurrent Cascade-Correlation, Elman nets, and Neural Sequence Chunking, LSTM leads to many more successful runs, and learns much faster. LSTM also solves complex, artificial long time lag tasks that have never been solved by previous recurrent network algorithms.

## 1 INTRODUCTION

Recurrent networks can in principle use their feedback connections to store representations of recent input events in form of activations (“short-term memory”, as opposed to “long-term memory” embodied by slowly changing weights). This is potentially significant for many applications, including speech processing, non-Markovian control, and music composition (e.g., Mozer 1992). The most widely used algorithms for learning *what* to put in short-term memory, however, take too much time or do not work well at all, especially when minimal time lags between inputs and corresponding teacher signals are long. Although theoretically fascinating, existing methods do not provide clear *practical* advantages over, say, backprop in feedforward nets with limited time windows. This paper will review an analysis of the problem and suggest a remedy.

**The problem.** With conventional “Back-Propagation Through Time” (BPTT, e.g., Williams and Zipser 1992, Werbos 1988) or “Real-Time Recurrent Learning” (RTRL, e.g., Robinson and Fallside 1987), error signals “flowing backwards in time” tend to either (1) blow up or (2) vanish: the temporal evolution of the backpropagated error exponentially depends on the size of the weights (Hochreiter 1991). Case (1) may lead to oscillating weights, while in case (2) learning to bridge long time lags takes a prohibitive amount of time, or does not work at all (see section 3).

**The remedy.** This paper presents “ *Long Short-Term Memory* ” (LSTM), a novel recurrent network architecture in conjunction with an appropriate gradient-based learning algorithm. LSTM is designed to overcome these error back-flow problems. It can learn to bridge time intervals in excess of 1000 steps even in case of noisy, incompressible input sequences, without loss of short time lag capabilities. This is achieved by an efficient, gradient-based algorithm for an architecture

1

where

$$ net\_{out\_j}(t) = \sum\_u w\_{out\_j u} y^u(t-1), $$

and

$$ net\_{in\_j}(t) = \sum\_u w\_{in\_j u} y^u(t-1). $$

We also have

$$ net\_{c\_j}(t) = \sum\_u w\_{c\_j u} y^u(t-1). $$

The summation indices $u$ may stand for input units, gate units, memory cells, or even conventional hidden units if there are any (see also paragraph on “network topology” below). All these different types of units may convey useful information about the current state of the net. For instance, an input gate (output gate) may use inputs from other memory cells to decide whether to store (access) certain information in its memory cell. There even may be recurrent self-connections like $w\_{c\_j c\_j}$. It is up to the user to define the network topology. See Figure 2 for an example.

At time $t$, $c\_j$’s output $y^{c\_j}(t)$ is computed as

$$ y^{c\_j}(t) = y^{out\_j}(t) h(s\_{c\_j}(t)), $$

where the “internal state” $s\_{c\_j}(t)$ is

$$ s\_{c\_j}(0) = 0, \quad s\_{c\_j}(t) = s\_{c\_j}(t-1) + y^{in\_j}(t) g(net\_{c\_j}(t)) \quad \text{for } t &gt; 0. $$

The differentiable function $g$ squashes $net\_{c\_j}$; the differentiable function $h$ scales memory cell outputs computed from the internal state $s\_{c\_j}$.

<!-- image -->

**VLM image description (pages: 2)**

This figure illustrates the architecture of a **Long Short-Term Memory (LSTM)** cell or a similar gated recurrent unit.

### Structure
*   **Input/Output:** The cell takes an input $\text{net}_{c_i}$ (weighted by $w_{c_i l}$) and produces an output $y^{c_i}$ (weighted by $w_{ic_j}$).
*   **Internal Components:**
    *   **Gates:** Represented by circles with sigmoid-like curves ($g$ and $h$).
    *   **State Update:** A central linear unit with a weight of $1.0$ representing the cell state memory.
    *   **External Inputs:** Additional inputs $\text{net}_{in_j}$ and $\text{net}_{out_j}$ modulate the flow via $y^{in_j}$ and $y^{out_j}$.
*   **Formula:** The state update is defined as:
    $$s_{c_i} = s_{c_i} + g y^{in_j}$$

### Key Takeaway
The diagram depicts a **gated memory mechanism**. The cell state ($s_{c_i}$) is updated additively, controlled by a gate ($g$) and an input signal ($y^{in_j}$), allowing the network to selectively remember or forget information over time to mitigate the vanishing gradient problem.


Figure 1: Architecture of memory cell $c\_j$ (the box) and its gate units $in\_j, out\_j$. The self-recurrent connection (with weight 1.0) indicates feedback with a delay of 1 time step. It builds the basis of the “constant error carousel” CEC. The gate units open and close access to CEC. See text and appendix A.1 for details.

**Why gate units?** To avoid input weight conflicts, $in\_j$ controls the error flow to memory cell $c\_j$’s input connections $w\_{c\_j i}$. To circumvent $c\_j$’s output weight conflicts, $out\_j$ controls the error flow from unit $j$’s output connections. In other words, the net can use $in\_j$ to decide when to keep or override information in memory cell $c\_j$, and $out\_j$ to decide when to access memory cell $c\_j$ and when to prevent other units from being perturbed by $c\_j$ (see Figure 1).

Error signals trapped within a memory cell’s CEC *cannot* change — but different error signals flowing into the cell (at different times) via its output gate may get superimposed. The output gate will have to learn *which* errors to trap in its CEC, by appropriately scaling them. The input

7

gate will have to learn when to release errors, again by appropriately scaling them. Essentially, the multiplicative gate units open and close access to constant error flow through CEC.

Distributed output representations typically do require output gates. Not always are both gate types necessary, though — one may be sufficient. For instance, in Experiments 2a and 2b in Section 5, it will be possible to use input gates only. In fact, output gates are not required in case of local output encoding — preventing memory cells from perturbing already learned outputs can be done by simply setting the corresponding weights to zero. Even in this case, however, output gates can be beneficial: they prevent the net’s attempts at storing long time lag memories (which are usually hard to learn) from perturbing activations representing easily learnable short time lag memories. (This will prove quite useful in Experiment 1, for instance.)

**Network topology.** We use networks with one input layer, one hidden layer, and one output layer. The (fully) self-connected hidden layer contains memory cells and corresponding gate units (for convenience, we refer to both memory cells and gate units as being located in the hidden layer). The hidden layer may also contain “conventional” hidden units providing inputs to gate units and memory cells. All units (except for gate units) in all layers have directed connections (serve as inputs) to all units in the layer above (or to all higher layers – Experiments 2a and 2b).

**Memory cell blocks.** $S$ memory cells sharing the same input gate and the same output gate form a structure called a “memory cell block of size $S$”. Memory cell blocks facilitate information storage — as with conventional neural nets, it is not so easy to code a distributed input within a single cell. Since each memory cell block has as many gate units as a single memory cell (namely two), the block architecture can be even slightly more efficient (see paragraph “computational complexity”). A memory cell block of size 1 is just a simple memory cell. In the experiments (Section 5), we will use memory cell blocks of various sizes.

**Learning.** We use a variant of RTRL (e.g., Robinson and Fallside 1987) which properly takes into account the altered, multiplicative dynamics caused by input and output gates. However, to ensure non-decaying error backpropagation, errors arriving at “memory cell net inputs” (for cell $c\_j$, this includes $net\_{c\_j}$, $net\_{in\_j}$, $net\_{out\_j}$) do not get propagated back further in time (although they *do* serve to change the incoming weights). Only within² memory cells, errors are propagated back through previous internal states $s\_{c\_j}$. To visualize this: once an error signal arrives at a memory cell output, it gets scaled by output gate activation and $h'$. Then it is within the memory cell’s CEC, where it can flow back indefinitely without ever being scaled. Only when it leaves the memory cell through the input gate and $g$, it is scaled once more by input gate activation and $g'$. It then serves to change the incoming weights before it is truncated (see appendix for explicit formulae).

**Computational complexity.** As with Mozer’s focused recurrent backprop algorithm (Mozer 1989), only the derivatives $\frac{\partial s\_{c\_j}}{\partial w\_{i1}}$ need to be stored and updated. Hence the LSTM algorithm is very efficient, with an excellent update complexity of $O(W)$, where $W$ the number of weights (see details in appendix A.1). Hence, LSTM and BPTT for fully recurrent nets have the same update complexity per time step (while RTRL’s is much worse). Unlike full BPTT, however, LSTM is *local in space and time* ³: there is no need to store activation values observed during sequence processing in a stack with potentially unlimited size.

**Abuse problem and solutions.** In the beginning of the learning phase, error reduction may be possible without storing information over time. The network will thus tend to abuse memory cells, e.g., as bias cells (i.e., it might make their activations constant and use the outgoing connections as adaptive thresholds for other units). The potential difficulty is: it may take a long time to release abused memory cells and make them available for further learning. A similar “abuse problem” appears if two memory cells store the same (redundant) information. There are at least two solutions to the abuse problem: (1) *Sequential network construction* (e.g., Fahlman 1991): a memory cell and the corresponding gate units are added to the network whenever the

²For intra-cellular backprop in a quite different context see also Doya and Yoshizawa (1989).

³Following Schmidhuber (1989), we say that a recurrent net algorithm is *local in space* if the update complexity per time step and weight does not depend on network size. We say that a method is *local in time* if its storage requirements do not depend on input sequence length. For instance, RTRL is local in space but not in time. BPTT is local in space but not in time.

8

Figure 2: Example of a net with 8 input units, 4 output units, and 2 memory cell blocks of size 2. in1 marks the input gate, out1 marks the output gate, and cell1/block1 marks the first memory cell of block 1. cell1/block1's architecture is identical to the one in Figure 1, with gate units in1 and out1 (note that by rotating Figure 1 by 90 degrees anti-clockwise, it will match with the corresponding parts of Figure 1). The example assumes dense connectivity: each gate unit and each memory cell see all non-output units. For simplicity, however, outgoing weights of only one type of unit are shown for each layer. With the efficient, truncated update rule, error flows only through connections to output units, and through fixed self-connections within cell blocks (not shown here - see Figure 1). Error flow is truncated once it "wants" to leave memory cells or gate units. Therefore, no connection shown above serves to propagate error back to the unit from which the connection originates (except for connections to output units), although the connections themselves are modifiable. That's why the truncated LSTM algorithm is so efficient, despite its ability to bridge very long time lags. See text and appendix A.1 for details. Figure 2 actually shows the architecture used for Experiment 6a - only the bias of the non-input units is omitted.

error stops decreasing (see Experiment 2 in Section 5). (2) Output gate bias: each output gate gets a negative initial bias, to push initial memory cell activations towards zero. Memory cells with more negative bias automatically get "allocated" later (see Experiments 1, 3, 4, 5, 6 in Section 5).

**Internal state drift and remedies.** If memory cell $c\_j$'s inputs are mostly positive or mostly negative, then its internal state $s\_j$ will tend to drift away over time. This is potentially dangerous, for the $h'(s\_j)$ will then adopt very small values, and the gradient will vanish. One way to circumvent this problem is to choose an appropriate function $h$. But $h(x) = x$, for instance, has the disadvantage of unrestricted memory cell output range. Our simple but effective way of solving drift problems at the beginning of learning is to initially bias the input gate $in\_j$ towards zero. Although there is a tradeoff between the magnitudes of $h'(s\_j)$ on the one hand and of $y^{in\_j}$ and $f\_{in\_j}^l$ on the other, the potential negative effect of input gate bias is negligible compared to the one of the drifting effect. With logistic sigmoid activation functions, there appears to be no need for fine-tuning the initial bias, as confirmed by Experiments 4 and 5 in Section 5.4.

## 5 EXPERIMENTS

**Introduction.** Which tasks are appropriate to demonstrate the quality of a novel long time lag

9

\begin{tabular}{|l|l|l|l|l|l|} \hline method &amp; hidden units &amp; # weights &amp; learning rate &amp; % of success &amp; success after \ \hline RTRL &amp; 3 &amp; $\approx$ 170 &amp; 0.05 &amp; `some fraction'' & 173,000 \\ RTRL & 12 & $\approx$ 494 & 0.1 &` some fraction'' &amp; 25,000 \ ELM &amp; 15 &amp; $\approx$ 435 &amp; &amp; 0 &amp; $&gt;$200,000 \ RCC &amp; 7-9 &amp; $\approx$ 119-198 &amp; &amp; 50 &amp; 182,000 \ LSTM &amp; 4 blocks, size 1 &amp; 264 &amp; 0.1 &amp; 100 &amp; 39,740 \ LSTM &amp; 3 blocks, size 2 &amp; 276 &amp; 0.1 &amp; 100 &amp; 21,730 \ LSTM &amp; 3 blocks, size 2 &amp; 276 &amp; 0.2 &amp; 97 &amp; 14,060 \ LSTM &amp; 4 blocks, size 1 &amp; 264 &amp; 0.5 &amp; 97 &amp; 9,500 \ LSTM &amp; 3 blocks, size 2 &amp; 276 &amp; 0.5 &amp; 100 &amp; 8,440 \ \hline \end{tabular}

Table 1: EXPERIMENT 1: Embedded Reber grammar: percentage of successful trials and number of sequence presentations until success for RTRL (results taken from Smith and Zipser 1989), `Elman net trained by Elman's procedure'' (results taken from Cleeremans et al. 1989),` Recurrent Cascade-Correlation'' (results taken from Fahlman 1991) and our new approach (LSTM). Weight numbers in the first 4 rows are estimates - the corresponding papers do not provide all the technical details. Only we almost always learns to solve the task (only two failures out of 150 trials). Even when we ignore the unsuccessful trials of the other approaches, LSTM learns much faster (the number of required training examples in the bottom row varies between 3,800 and 24,100).

tations. A successful run is one that fulfills the following criterion: after training, during 10,000 successive, randomly chosen input sequences, the maximal absolute error of all output units is always below 0.25.

**Architectures.** RTRL: one self-recurrent hidden unit, $p+1$ non-recurrent output units. Each layer has connections from all layers below. All units use the logistic activation function sigmoid in [0,1].

BPTT: same architecture as the one trained by RTRL.

CH: both net architectures like RTRL's, but one has an additional output for predicting the hidden unit of the other one (see Schmidhuber 1992b for details).

LSTM: like with RTRL, but the hidden unit is replaced by a memory cell and an input gate (no output gate required). $g$ is the logistic sigmoid, and $h$ is the identity function $h : h(x) = x, \forall x$. Memory cell and input gate are added once the error has stopped decreasing (see abuse problem: solution (1) in Section 4).

**Results.** Using RTRL and a short 4 time step delay ($p = 4$), $\frac{7}{9}$ of all trials were successful. *No trial was successful with $p = 10$.* With *long* time lags, only the neural sequence chunker and LSTM achieved successful trials, while BPTT and RTRL failed. With $p = 100$, the 2-net sequence chunker solved the task in only $\frac{1}{3}$ of all trials. LSTM, however, always learned to solve the task. Comparing successful trials only, LSTM learned much faster. See Table 2 for details. It should be mentioned, however, that a *hierarchical* chunker can also always quickly solve this task (Schmidhuber 1992c, 1993).

**Task 2b: no local regularities.** With the task above, the chunker sometimes learns to correctly predict the final element, but only because of predictable local regularities in the input stream that allow for compressing the sequence. In an additional, more difficult task (involving many more different possible sequences), we remove compressibility by replacing the deterministic subsequence $(a\_1, a\_2, \dots, a\_{p-1})$ by a *random* subsequence (of length $p-1$) over the alphabet $a\_1, a\_2, \dots, a\_{p-1}$. We obtain 2 classes (two sets of sequences) $ { (y, a\_{i\_1}, a\_{i\_2}, \dots, a\_{i\_{p-1}}, y) \mid 1 \le i\_1, i\_2, \dots, i\_{p-1} \le p-1 } $ and $ { (x, a\_{i\_1}, a\_{i\_2}, \dots, a\_{i\_{p-1}}, x) \mid 1 \le i\_1, i\_2, \dots, i\_{p-1} \le p-1 } $. Again, every next sequence element has to be predicted. The only totally predictable targets, however, are $x$ and $y$, which occur at sequence ends. Training exemplars are chosen randomly from the 2 classes. Architectures and parameters are the same as in Experiment 2a. A successful run is one that fulfills the following criterion: after training, during 10,000 successive, randomly chosen input

13

\begin{tabular}{|l|l|l|l|l|l|} \hline Method &amp; Delay $p$ &amp; Learning rate &amp; # weights &amp; % Successful trials &amp; Success after \ \hline RTRL &amp; 4 &amp; 1.0 &amp; 36 &amp; 78 &amp; 1,043,000 \ RTRL &amp; 4 &amp; 4.0 &amp; 36 &amp; 56 &amp; 892,000 \ RTRL &amp; 4 &amp; 10.0 &amp; 36 &amp; 22 &amp; 254,000 \ RTRL &amp; 10 &amp; 1.0-10.0 &amp; 144 &amp; 0 &amp; $&gt; 5,000,000$ \ RTRL &amp; 100 &amp; 1.0-10.0 &amp; 10404 &amp; 0 &amp; $&gt; 5,000,000$ \ BPTT &amp; 100 &amp; 1.0-10.0 &amp; 10404 &amp; 0 &amp; $&gt; 5,000,000$ \ CH &amp; 100 &amp; 1.0 &amp; 10506 &amp; 33 &amp; 32,400 \ LSTM &amp; 100 &amp; 1.0 &amp; 10504 &amp; 100 &amp; 5,040 \ \hline \end{tabular}

Table 2: Task 2a: Percentage of successful trials and number of training sequences until success, for "Real-Time Recurrent Learning" (RTRL), "Back-Propagation Through Time" (BPTT), neural sequence chunking (CH), and the new method (LSTM). Table entries refer to means of 18 trials. With 100 time step delays, only CH and LSTM achieve successful trials. Even when we ignore the unsuccessful trials of the other approaches, LSTM learns much faster.

sequences, the maximal absolute error of all output units is below 0.25 at sequence end.

**Results.** As expected, the chunker failed to solve this task (so did BPTT and RTRL, of course). LSTM, however, was always successful. On average (mean of 18 trials), success for $p = 100$ was achieved after 5,680 sequence presentations. This demonstrates that LSTM does not require sequence regularities to work well.

**Task 2c: very long time lags - no local regularities.** This is the most difficult task in this subsection. To our knowledge no other recurrent net algorithm can solve it. Now there are $p+4$ possible input symbols denoted $a\_1, ..., a\_{p-1}, a\_p, a\_{p+1} = e, a\_{p+2} = b, a\_{p+3} = x, a\_{p+4} = y$. $a\_1, ..., a\_p$ are also called "distractor symbols". Again, $a\_i$ is locally represented by the $p+4$-dimensional vector whose $i$th component is 1 (all other components are 0). A net with $p+4$ input units and 2 output units sequentially observes input symbol sequences, one at a time. Training sequences are randomly chosen from the union of two very similar subsets of sequences: $ { (b, y, a\_{i\_1}, a\_{i\_2}, \dots, a\_{i\_{q+k}}, e, y) \mid 1 \le i\_1, i\_2, \dots, i\_{q+k} \le q } $ and $ { (b, x, a\_{i\_1}, a\_{i\_2}, \dots, a\_{i\_{q+k}}, e, x) \mid 1 \le i\_1, i\_2, \dots, i\_{q+k} \le q } $. To produce a training sequence, we (1) randomly generate a sequence prefix of length $q+2$, (2) randomly generate a sequence suffix of additional elements ($\ne b, e, x, y$) with probability $\frac{9}{10}$ or, alternatively, an $e$ with probability $\frac{1}{10}$. In the latter case, we (3) conclude the sequence with $x$ or $y$, depending on the second element. For a given $k$, this leads to a uniform distribution on the possible sequences with length $q+k+4$. The minimal sequence length is $q+4$; the expected length is

$$ 4 + \sum\_{k=0}^{\infty} \frac{1}{10} \left(\frac{9}{10}\right)^k (q+k) = q + 14. $$

The expected number of occurrences of element $a\_i, 1 \le i \le p$, in a sequence is $\frac{q+10}{p} \approx \frac{q}{p}$. The goal is to predict the last symbol, which always occurs after the "trigger symbol" $e$. Error signals are generated only at sequence ends. To predict the final element, the net has to learn to store a representation of the second element for at least $q+1$ time steps (until it sees the trigger symbol $e$). Success is defined as "prediction error (for final sequence element) of both output units always below 0.2, for 10,000 successive, randomly chosen input sequences".

**Architecture/Learning.** The net has $p+4$ input units and 2 output units. Weights are initialized in [-0.2,0.2]. To avoid too much learning time variance due to different weight initializations, the hidden layer gets two memory cells (two cell blocks of size 1 - although one would be sufficient). There are no other hidden units. The output layer receives connections only from memory cells. Memory cells and gate units receive connections from input units, memory cells and gate units (i.e., the hidden layer is fully connected). No bias weights are used. $h$ and $g$ are logistic sigmoids with output ranges [-1,1] and [-2,2], respectively. The learning rate is 0.01.

14