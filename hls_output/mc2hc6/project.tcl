variable project_name
set project_name "mc2hc6"
variable backend
set backend "vitis"
variable part
set part "xc7z020clg400-1"
variable clock_period
set clock_period 10
variable version
set version "1.0.0"

# HLS build script for $project_name with 64-bit AXIS wrapper top
# Works on classic Vitis HLS CLI; for Vitis 2024/2025 unified, use v++ with --hls.pre_tcl to source this file.

set src_dir "/home/makin/development/fpga/source/cnnmakin/makin_cifar/hls_output/$project_name/firmware"

# membuat folder ./$project_name
open_project $project_name
set_top mc2h_axis64

add_files "$src_dir/mc2h_axis_wrapper.cpp"
add_files "$src_dir/${project_name}.cpp"
# Optionally add other helper sources if present
# add_files "$src_dir/weights.cpp"

# Testbench
add_files -tb "$src_dir/tb_mc2h_axis64.cpp"

# membuat folder ./$project_name/sol1, IP disitu
open_solution sol1
set_part $part
create_clock -period $clock_period -name default

# C-simulation (pass path to test image)
catch { csim_design -argv "${src_dir}/../tb_data/X_test2.npy" }

# Synthesis with parallel jobs
csynth_design

# RTL co-sim, without waveform tracing (fastest accurate check)
catch { cosim_design -rtl verilog -tool xsim -argv "${src_dir}/../tb_data/X_test2.npy" }
export_design -format ip_catalog -ipname ${project_name}_axis64

exit
