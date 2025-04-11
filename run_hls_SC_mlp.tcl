# Create a new Vitis HLS project
open_project -reset GPT2_block_0_SC_mlp
set_top mlp_block_forward 
add_files ./llmc/dataloader.cpp 
add_files ./llmc/rand.cpp 
add_files ./llmc/tokenizer.cpp 
add_files ./llmc/utils.cpp
add_files ./llmc/SC.cpp
add_files train_gpt2.cpp -cflags "-D TESTING_MLP -D SC_MATMUL" 

add_files -tb test_gpt2_mlp.cpp

# Add data files (for simulation)
add_files -tb gpt2_124M.bin
add_files -tb gpt2_124M_debug_state.bin
add_files -tb gpt2_124M_block_0_mlp_debug_state.bin

# Create a solution
open_solution -reset "solution_GPT2_block_0_SC_mlp"
set_part xcu280-fsvh2892-2LV-e ;# Set the target FPGA part (modify as needed)

# Run C simulation
# csim_design

# Run High-Level Synthesis (HLS)
csynth_design

# Run co-simulation
# cosim_design

# Export RTL
# export_design -format ip_catalog
exit
