# wayfinding_vision

To fix noetic-gpu dockerfile (make sure you are on noetic branch):
```
RUN rm /etc/apt/sources.list.d/cuda.list
# RUN rm /etc/apt/sources.list.d/nvidia-ml.list
```
Also change `QObject` to `: public QObject` to `class _AdditionalTopicSubscriber` in `additional_topic_subscriber.h` from `spencer_tracking_rviz_plugin` to fix build error.

To build with no cached docker container layers, add `--no-cache` to the Makefile.

The camera output needs to be changed to:
`ros2 launch realsense2_camera rs_launch.py camera_name:=rgbd_front_top camera_namespace:=/spencer/sensors align_depth.enable:=true tf_publish_rate:=30.0`.
Since it needs to match `/spencer/sensors/rgbd_front_top`.

Changes to `run.sh` (not sure if necessary?):
```
# Parse optional command-line arguments
USE_HOST_NETWORK=1
while :; do
    case $1 in -h|--host-network) USE_HOST_NETWORK=1; shift;
        ;;
        *) break
    esac
done
EXTRA_DOCKER_ARGS="$@"

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


XAUTH=/tmp/.docker.xauth
touch $XAUTH
xauth nlist :1 | sed -e 's/^..../ffff/' | xauth -f $XAUTH nmerge -
chmod a+r $XAUTH

# Run the Docker container in interactive mode
echo "starting container..."
            # --env="QT_X11_NO_MITSHM=1" \
            # --env="XAUTHORITY=$XAUTH" \
            # --volume="$XAUTH:$XAUTH" \
docker run --rm -it \
            --env="DISPLAY=$DISPLAY" \
            --volume="/tmp/.X11-unix:/tmp/.X11-unix" \
            $RUNTIME_ARGS \
            $NETWORK_ARGS \
            $EXTRA_DOCKER_ARGS \
            --env="QT_X11_NO_MITSHM=1" \
            --env="XAUTHORITY=$XAUTH" \
            --volume="$XAUTH:$XAUTH" \
            spencer/spencer_people_tracking:$IMAGE_VARIANT \
            $COMMAND
```
