copyright (c) Izzulmakin, 2025

### Steps

- s0datasets.py for building dataset
- s1modelc10.py for building model with cifar10. will generate keras files in keras/ 
- s1modelc2.py for building model with cifar2 (2 labels: animal/transport). this can be ignored.
- s2hlsmodelZ2c10.py for building hlsmodel, generate hls files in hls_output/
- s3hls.py automate these steps:
	- copy **mc2h_axis_wrapper.*, tb_mc2h_axis64.cpp, project.tcl** to the newly generated folder in `hls_output/{outputfoldername}/firmware` and `hls_output/{outputfoldername}/`
	- edit mc2h_axis_wrapper.* to include & call the correct "top function" in `hls_output/{outputfoldername}`
	- update `hls_output/{outputfoldername}/firmware/parameters.h` to hardcode WEIGHTS_DIR like:
		`#define WEIGHTS_DIR "/absolute/path/to/hls_output/mc2hc9/firmware/weights"`
	- run `cd hls_output/{outputfoldername}` && `vitis_hls -f project.tcl`
- **NOTE** check img/{outputfoldername}.jpg for detailed quantization, and make adjustment for mc2h_axis_wrapper.* and s6pynqtestc10.py based on output quantization!
- check latency in `hls_output/{outputfoldername}/{outputfoldername}/sol1/syn/report/mc2h_axis64_csynth.rpt` Latency -> Detail -> Instance
- use exported IP `hls_output/{outputfoldername}/{outputfoldername}/sol1/impl/ip` in vivado with design like

