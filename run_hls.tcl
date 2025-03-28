# Create a new Vitis HLS project
open_project GPT2_inference
set_top gpt2_forward
add_files ./llmc/dataloader.c
add_files ./llmc/rand.c
add_files ./llmc/tokenizer.c
add_files ./llmc/utils.c
add_files train_gpt2.c -cflags "-D HLS_CSIM" 

add_files -tb test_gpt2.c

# Add data files (for simulation)
add_files -tb gpt2_124M.bin
add_files -tb gpt2_124M_debug_state.bin

# Create a solution
open_solution "solution_GPT2_inference"
set_part xcu280-fsvh2892-2LV-e ;# Set the target FPGA part (modify as needed)

# Run C simulation
csim_design

# Run High-Level Synthesis (HLS)
# csynth_design

# Run co-simulation
# cosim_design

# Export RTL
# export_design -format ip_catalog
exit
