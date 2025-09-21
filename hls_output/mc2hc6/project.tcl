# variable project_name
# set project_name "mc2hc6c6"
# variable backend
# set backend "vitis"
# variable part
# set part "xc7z020clg400-1"
# variable clock_period
# set clock_period 5
# variable clock_uncertainty
# set clock_uncertainty 27%
# variable version
# set version "1.0.0"
# variable maximum_size
# set maximum_size 4096

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

# HLS build script for mc2hc6 with 64-bit AXIS wrapper top
# Works on classic Vitis HLS CLI; for Vitis 2024/2025 unified, use v++ with --hls.pre_tcl to source this file.

set src_dir "/home/makin/development/fpga/source/cnnmakin/makin_cifar/hls_output/$project_name/firmware"

# membuat folder ./mc2hc6
open_project $project_name
set_top mc2h_axis64

add_files "$src_dir/mc2h_axis_wrapper.cpp"
add_files "$src_dir/${project_name}.cpp"
# Optionally add other helper sources if present
# add_files "$src_dir/weights.cpp"

# Testbench
add_files -tb "$src_dir/tb_mc2h_axis64.cpp"

# membuat folder ./mc2hc6/sol1, IP disitu
open_solution sol1
set_part $part
create_clock -period $clock_period -name default

# C-simulation (provide npy via working dir/file)
catch { csim_design }

# Synthesis with parallel jobs
csynth_design -jobs 6

# Optionally co-sim and export IP with parallel jobs
# catch { cosim_design -rtl verilog -jobs 6 }
export_design -format ip_catalog -ipname ${project_name}_axis64

exit
