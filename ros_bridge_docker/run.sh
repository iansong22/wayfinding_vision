#!/bin/bash

if [ $# -eq 0 ]
then
    IMAGE_VARIANT="foxy"
else
    # Get name of Docker image
    IMAGE_VARIANT=$1
    shift
fi
echo "Variant of Docker image set to: $IMAGE_VARIANT"

# Parse optional command-line arguments
USE_HOST_NETWORK=1
while :; do
    case $1 in -h|--host-network) USE_HOST_NETWORK=1; shift;
        ;;
        *) break
    esac
done

# Build network args
if [ $USE_HOST_NETWORK -eq 1 ]
then
    # This currently requires privileged access, otherwise rviz will crash. See:
    # https://answers.ros.org/question/301056/ros2-rviz-in-docker-container/?answer=301062#post-id-301062
    NETWORK_ARGS="--network=host --privileged --ipc=host --pid=host";
    echo "connection to host network enabled, running in privileged mode"
else
    NETWORK_ARGS="";
    echo "connection to host network disabled, network will be isolated from host. use -h to change this"
fi

# Following lines are based upon code from the ROS wiki
# They ensure that X server can be used for graphical applications, e.g. rviz
# See http://wiki.ros.org/docker/Tutorials/Hardware%20Acceleration#nvidia-docker2
# Copyright 2020 Open Source Robotics Foundation (CC-BY 3.0 license)
XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist :1 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod a+r $XAUTH

# Run the Docker container in interactive mode
echo "starting container..."
            # --env="QT_X11_NO_MITSHM=1" \
            # --env="XAUTHORITY=$XAUTH" \
            # --volume="$XAUTH:$XAUTH" \
docker run -it --rm \
            --env="DISPLAY=$DISPLAY" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" \
            $NETWORK_ARGS \
            -v /dev/shm:/dev/shm \
            --env="QT_X11_NO_MITSHM=1" \
            --env ROS_DOMAIN_ID=0 \
            --env RMW_IMPLEMENTATION=rmw_cyclonedds_cpp \
            -v ./src/CycloneDDS.xml:/root/CycloneDDS.xml \
            wayfinding/ros1_bridge:$IMAGE_VARIANT
            # --env CYCLONEDDS_URI=file:///root/CycloneDDS.xml \
            
            
