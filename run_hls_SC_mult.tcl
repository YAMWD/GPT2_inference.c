# Create a new Vitis HLS project
open_project -reset SC_mult
set_top SC_mult 
# add_files ./llmc/dataloader.c 
# add_files ./llmc/rand.c 
# add_files ./llmc/tokenizer.c 
# add_files ./llmc/utils.c 
add_files ./llmc/SC.cpp
# add_files train_gpt2.cpp -cflags "-D TESTING_ATTN" 

add_files -tb test_SC_mult.cpp

# Add data files (for simulation)
add_files -tb gpt2_124M.bin
add_files -tb gpt2_124M_debug_state.bin

# Create a solution
open_solution -reset "solution_SC_mult"
set_part xcu280-fsvh2892-2LV-e ;# Set the target FPGA part (modify as needed)

# Run C simulation
csim_design

# Run High-Level Synthesis (HLS)                                                                                            nnn 
# csynth_design

# Run co-simulation
# cosim_design

# Export RTL
# export_design -format ip_catalog
exit
