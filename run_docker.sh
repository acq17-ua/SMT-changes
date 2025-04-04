docker run --rm -it --gpus all \
	-e DISPLAY=${DISPLAY} \
    	-v /tmp/.X11-unix:/tmp/.X11-unix:rw \
	--network host \
	--workdir="/workspace" \
    	--name smt-testing \
	--runtime=nvidia \
	-e NVIDIA_VISIBLE_DEVICES=all -e NVIDIA_DRIVER_CAPABILITIES=all \
	-e "TERM=xterm-256color" \
    	--volume="$PWD:/workspace:rw" \
    	smt-testing:latest bash
