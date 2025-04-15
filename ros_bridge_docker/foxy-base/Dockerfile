FROM osrf/ros:foxy-desktop

ENV ROS_DISTRO=noetic
ENV ROS2_DISTRO=foxy

# Install ROS1 Noetic
RUN curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add -
    RUN echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros1.list
    RUN apt-get update && apt-get install -y \
        ros-noetic-desktop \
        python3-rosdep \
        python3-rosinstall \
        python3-rosinstall-generator \
        python3-wstool \
        build-essential

# RUN echo "source /opt/ros/noetic/setup.bash" >> /root/.bashrc
# RUN echo "source /opt/ros/foxy/setup.bash" >> /root/.bashrc

RUN apt-get install -y git

RUN apt-get install --no-install-recommends --quiet --yes \
			build-essential \
			clang \
			cmake \
			git \
			lcov \
			libasio-dev \
			libc++-dev \
			libc++abi-dev \
			libssl-dev \
			libtinyxml2-dev \
			python3-dev \
			python3-pip \
			python3-vcstool \
			python3-wheel \
			python3-rosdep \
			wget
# RUN pip3 install --upgrade \
# 			argcomplete \
# 			catkin_pkg \
# 			colcon-bash==0.4.2 \
# 			colcon-cd==0.1.1 \
# 			colcon-cmake==0.2.22 \
# 			colcon-common-extensions==0.2.1 \
# 			colcon-core==0.15.1 \
# 			colcon-coveragepy-result==0.0.8 \
# 			colcon-defaults==0.2.7 \
# 			colcon-lcov-result==0.5.0 \
# 			colcon-library-path==0.2.1 \
# 			colcon-metadata==0.2.5 \
# 			colcon-mixin==0.2.2 \
# 			colcon-notification==0.2.15 \
# 			colcon-output==0.2.12 \
# 			colcon-package-information==0.3.3 \
# 			colcon-package-selection==0.2.10 \
# 			colcon-parallel-executor==0.2.4 \
# 			colcon-pkg-config==0.1.0 \
# 			colcon-powershell==0.3.7 \
# 			colcon-python-setup-py==0.2.7 \
# 			colcon-recursive-crawl==0.2.1 \
# 			colcon-ros==0.3.23 \
# 			colcon-test-result==0.3.8 \
# 			coverage \
# 			cryptography \
# 			"empy<4" \
# 			"flake8<3.8" \
# 			flake8-blind-except \
# 			flake8-builtins \
# 			flake8-class-newline \
# 			flake8-comprehensions \
# 			flake8-deprecated \
# 			flake8-docstrings \
# 			flake8-import-order \
# 			flake8-quotes \
# 			ifcfg \
# 			importlib-metadata==2.* \
# 			lark-parser \
# 			mock \
# 			mypy \
# 			nose \
# 			pep8 \
# 			pydocstyle \
# 			pytest \
# 			pytest-cov \
# 			pytest-mock \
# 			pytest-repeat \
# 			pytest-rerunfailures \
# 			pytest-runner \
# 			setuptools==58.2.0 \
# 			pyparsing \
# 			wheel
# RUN /bin/bash -c 'source /opt/ros/noetic/setup.bash && catkin_make'
# RUN /bin/bash -c 'source /opt/ros/foxy/setup.bash && colcon build'

ENV COLCON_WS=/ros_workspace
RUN mkdir -p $COLCON_WS/src

# Setup workspace
WORKDIR $COLCON_WS/src

RUN git clone -b $ROS2_DISTRO https://github.com/ros2/ros1_bridge.git
WORKDIR $COLCON_WS/src/ros1_bridge

RUN apt-get update && apt-get install -y \
    ros-foxy-rmw-cyclonedds-cpp \
    ros-foxy-demo-nodes-cpp \
    ros-foxy-demo-nodes-py && rm -rf /var/lib/apt/lists/*
ENV RMW_IMPLEMENTATION="rmw_cyclonedds_cpp"
RUN /bin/bash -c "colcon build --symlink-install --packages-skip ros1_bridge"

RUN /bin/bash -c "source /opt/ros/noetic/setup.bash"
RUN /bin/bash -c "source /opt/ros/foxy/setup.bash"


RUN /bin/bash -c "colcon build --symlink-install --packages-select ros1_bridge --cmake-force-configure"

# Entrypoint
CMD ["/bin/bash"]
