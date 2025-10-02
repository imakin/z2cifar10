import os
from sys import argv
from shutil import copy
from subprocess import check_output as co
from subprocess import Popen as po
import argparse


def exec(cmd):
    return co(cmd, shell=True).decode('utf8')
def exec2(*args, **kwargs):
    p = po(*args, **kwargs)



parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, required=True, default=None)
parser.add_argument('--novitis', action='store_true', default=False)
args = parser.parse_args()

target = args.name
c10 = True
if target.startswith('mc2'):
    c10 = False

# make sure chdir ke file ini
BASE_DIR = os.path.dirname(os.path.realpath(__file__))
os.chdir(BASE_DIR)




specific_source = "mc2hc6"
if c10:
    specific_source = "mc10c8"

copy(f'hls_output/{specific_source}/project.tcl', f'hls_output/{target}/project.tcl')
# copy(f'hls_output/{specific_source}/tb_data/X_test2.npy', f'hls_output/{target}/tb_data/X_test2.npy') # now use npy/*_X_test_main.npy
copy(f'hls_output/{specific_source}/firmware/tb_mc2h_axis64.cpp', f'hls_output/{target}/firmware/tb_mc2h_axis64.cpp')
copy(f'hls_output/{specific_source}/firmware/mc2h_axis_wrapper.h', f'hls_output/{target}/firmware/mc2h_axis_wrapper.h')
copy(f'hls_output/{specific_source}/firmware/mc2h_axis_wrapper.cpp', f'hls_output/{target}/firmware/mc2h_axis_wrapper.cpp')

# edit "hls_output/{target}/firmware/parameters.h",
# insert di baris ketiga f'#define WEIGHTS_DIR "{BASE_DIR}/hls_output/{target}/firmware/weights"'
with open(f'hls_output/{target}/firmware/parameters.h', 'r') as f:
    lines = f.readlines()
lines.insert(2, f'#define WEIGHTS_DIR "{BASE_DIR}/hls_output/{target}/firmware/weights"\n')
with open(f'hls_output/{target}/firmware/parameters.h', 'w') as f:
    f.writelines(lines)

# edit "hls_output/{target}/firmware/mc2h_axis_wrapper.cpp",
# replace all "[a-z0-9]+(core_in, core_out);" with "{target}(core_in, core_out);"
import re
with open(f'hls_output/{target}/firmware/mc2h_axis_wrapper.cpp', 'r') as f:
    content = f.read()
content = re.sub(r'[a-z0-9]+\(core_in, core_out\);', f'{target}(core_in, core_out);', content)
with open(f'hls_output/{target}/firmware/mc2h_axis_wrapper.cpp', 'w') as f:
    f.write(content)

# edit "hls_output/{target}/firmware/mc2h_axis_wrapper.h",
# replace all '#include "[a-z0-9]+.h"' with '#include "{target}.h"'
with open(f'hls_output/{target}/firmware/mc2h_axis_wrapper.h', 'r') as f:
    content = f.read()
content = re.sub(r'#include "[a-z0-9]+.h"', f'#include "{target}.h"', content)
with open(f'hls_output/{target}/firmware/mc2h_axis_wrapper.h', 'w') as f:
    f.write(content)


# edit "hls_output/{target}/project.tcl",
# replace line2 set project_name "mc2hc6" to set project_name "{target}"
with open(f'hls_output/{target}/project.tcl', 'r') as f:
    lines = f.readlines()
lines[1] = f'set project_name "{target}"\n'
with open(f'hls_output/{target}/project.tcl', 'w') as f:
    f.writelines(lines)

# edit "hls_output/{target}/project.tcl",
# replace baris yang berawalan set src_dir ke 'set src_dir "{BASE_DIR}/hls_output/$project_name/firmware"'
with open(f'hls_output/{target}/project.tcl', 'r') as f:
    lines = f.readlines()
for i in range(len(lines)):
    if lines[i].startswith('set src_dir'):
        lines[i] = f'set src_dir "{BASE_DIR}/hls_output/$project_name/firmware"\n'
with open(f'hls_output/{target}/project.tcl', 'w') as f:
    f.writelines(lines)

print(f'Finished setting up HLS project for target: {target}')
os.chdir(f'hls_output/{target}')
if not args.novitis:
    print('Running Vitis HLS...')
    os.system('vitis-run --mode hls --tcl project.tcl')