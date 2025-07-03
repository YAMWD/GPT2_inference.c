# Stochastic Computing GPT-2

This is the repo for Stochastic Computing GPT-2 based on karpathy's open source project llm.c.

Different branches implements different SC MAC.

The ```SC``` branch contains the full SC-GPT-2 implementation based on the most BISC-MAC.

The ```SC_matmul``` branch contains the BISC-MAC implementation.

The ```SC_MATMUL_HALTON``` branch contains the BISC-MAC implementation using Halton sequence.

The ```bipolar_APC_SC_MAC``` branch contains the APC-based SC-MAC.

Each branch contains TCL scripts that can be executed to verify the correctness of the corresponding implementation.

For example, by runnnig

```vitis_hls -f run_hls_SC_c_fc.tcl```

It performs the synthesis of the SC-based c_fc layer from the first block of the model.

To run on hardware:

```make run TARGET=hw```
