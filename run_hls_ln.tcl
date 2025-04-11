# Create a new Vitis HLS project
open_project GPT2_block_0_ln_0
set_top layernorm_forward 
add_files ./llmc/dataloader.cpp
add_files ./llmc/rand.cpp
add_files ./llmc/tokenizer.cpp
add_files ./llmc/utils.cpp
add_files train_gpt2.cpp -cflags "-D TESTING_LN" 

add_files -tb test_gpt2_ln.c

# Add data files (for simulation)
add_files -tb gpt2_124M.bin
add_files -tb gpt2_124M_debug_state.bin
add_files -tb gpt2_124M_block_0_ln_0_debug_state.bin

# Create a solution
open_solution "solution_GPT2_block_0_ln_0"
set_part xcu280-fsvh2892-2LV-e ;# Set the target FPGA part (modify as needed)

# Run C simulation
csim_design

# Run High-Level Synthesis (HLS)
csynth_design

# Run co-simulation
cosim_design

# Export RTL
# export_design -format ip_catalog
exit
