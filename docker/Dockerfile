FROM nvcr.io/nvidia/deepstream:7.1-triton-multiarch

ARG CMAKE_VERSION_MAJOR=3
ARG CMAKE_VERSION_MINOR=25
ARG CMAKE_VERSION_PATCH=3

ARG EIGEN_VERSION_MAJOR=3
ARG EIGEN_VERSION_MINOR=4
ARG EIGEN_VERSION_PATCH=0

ARG OPENCV_VERSION_MAJOR=4
ARG OPENCV_VERSION_MINOR=11
ARG OPENCV_VERSION_PATCH=0

ARG PCL_VERSION_MAJOR=1
ARG PCL_VERSION_MINOR=10
ARG PCL_VERSION_PATCH=0

ARG PYBIND11_VERSION_MAJOR=2
ARG PYBIND11_VERSION_MINOR=13
ARG PYBIND11_VERSION_PATCH=0

ARG YAML_CPP_VERSION_MAJOR=0
ARG YAML_CPP_VERSION_MINOR=8
ARG YAML_CPP_VERSION_PATCH=0

ENV TZ=US/Pacific
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install dependencies
RUN apt-get update --fix-missing && apt-get install -y --no-install-recommends \
    # Python and development tools
    python3-pip \
    python3-dev \
    # Build tools and compilers
    build-essential \
    cmake \
    cmake-curses-gui \
    checkinstall \
    g++ \
    gcc \
    gfortran \
    # Version control and utilities
    git \
    vim \
    tmux \
    wget \
    curl \
    bzip2 \
    ca-certificates \
    gnupg \
    software-properties-common \
    # Graphics and GUI libraries
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgtk2.0-dev \
    qtbase5-dev \
    # Math and science libraries
    libblas-dev \
    liblapack-dev \
    libatlas-base-dev \
    # Security and networking
    libssl-dev \
    libzmq3-dev \
    # Boost libraries
    libboost-filesystem-dev \
    libboost-date-time-dev \
    libboost-iostreams-dev \
    libboost-system-dev \
    libboost-program-options-dev \
    libboost-all-dev \
    # Point cloud and 3D processing
    libflann-dev \
    # Image and video processing
    libjpeg8-dev \
    libtiff5-dev \
    pkg-config \
    yasm \
    libavcodec-dev \
    libavformat-dev \
    libswscale-dev \
    libdc1394-dev \
    libxine2-dev \
    libv4l-dev \
    libtbb-dev \
    ffmpeg \
    # Audio/video codecs
    libfaac-dev \
    libmp3lame-dev \
    libtheora-dev \
    libvorbis-dev \
    libxvidcore-dev \
    libopencore-amrnb-dev \
    libopencore-amrwb-dev \
    x264 \
    v4l-utils \
    # Protocol buffers and logging
    libprotobuf-dev \
    protobuf-compiler \
    libgoogle-glog-dev \
    libgflags-dev \
    # Additional libraries
    libgphoto2-dev \
    libhdf5-dev \
    doxygen \
    proj-data \
    libproj-dev \
    libyaml-cpp-dev \
    libzmq3-dev \
    freeglut3-dev \
    libopenexr-dev \
    && rm -rf /var/lib/apt/lists/*

# Use gcc-12 to avoid gcc-11 internal compiler errors when building OpenCV 4.11.
# librealsense source list (preinstalled in deepstream base) has an expired GPG key
# and is unused here, so drop it before apt update.
RUN rm -f /etc/apt/sources.list.d/*librealsense*.list && \
    apt-get update && \
    apt-get install -y --no-install-recommends gcc-12 g++-12 && \
    update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-12 100 \
                        --slave  /usr/bin/g++ g++ /usr/bin/g++-12 && \
    rm -rf /var/lib/apt/lists/*


# Install cmake
RUN cd / &&\
wget http://www.cmake.org/files/v${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}/cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
tar xf cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH}.tar.gz &&\
cd cmake-${CMAKE_VERSION_MAJOR}.${CMAKE_VERSION_MINOR}.${CMAKE_VERSION_PATCH} &&\
./configure &&\
make &&\
make install


# Install Eigen
RUN cd / && \
    wget https://gitlab.com/libeigen/eigen/-/archive/${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}/eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    tar xf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz && \
    cd eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH} && \
    mkdir build && \
    cd build && \
    cmake .. && \
    make install && \
    cd / && \
    rm -rf eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}.tar.gz eigen-${EIGEN_VERSION_MAJOR}.${EIGEN_VERSION_MINOR}.${EIGEN_VERSION_PATCH}


# Install OpenCV
RUN cd / && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv && \
    git clone --depth 1 --branch ${OPENCV_VERSION_MAJOR}.${OPENCV_VERSION_MINOR}.${OPENCV_VERSION_PATCH} https://github.com/opencv/opencv_contrib && \
    mkdir -p /opencv/build && \
    cd /opencv/build && \
    cmake ..  -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_CUDA_STUBS=OFF \
        -DBUILD_DOCS=OFF \
        -DWITH_MATLAB=OFF \
        -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF \
        -DCUDA_FAST_MATH=ON \
        -DMKL_WITH_OPENMP=ON \
        -DOPENCV_ENABLE_NONFREE=ON \
        -DWITH_OPENMP=ON \
        -DWITH_QT=ON \
        -DWITH_OPENEXR=ON \
        -DBUILD_OPENEXR=OFF \
        -DENABLE_PRECOMPILED_HEADERS=OFF \
        -DBUILD_opencv_cudacodec=OFF \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DWITH_TIFF=OFF \
        -DWITH_WEBP=OFF \
        -DWITH_FFMPEG=ON \
        -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DENABLE_CXX11=OFF \
        -DBUILD_opencv_xfeatures2d=OFF \
        -DOPENCV_DNN_OPENCL=OFF \
        -DWITH_CUDA=ON \
        -DWITH_OPENCL=OFF \
        -DBUILD_opencv_wechat_qrcode=OFF \
        -DCMAKE_CXX_STANDARD=17 \
        -DCMAKE_CXX_STANDARD_REQUIRED=ON \
        -DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
        -DCUDA_ARCH_BIN="8.9" \
        -DBUILD_LIST=core,imgproc,imgcodecs,highgui,calib3d,features2d,rgbd,cudaarithm,cudafeatures2d,cudafilters,cudaimgproc,cudaoptflow,cudawarping,cudev,python3 \
        -DBUILD_opencv_apps=OFF \
        -DBUILD_TESTS=OFF \
        -DBUILD_PERF_TESTS=OFF \
        -DBUILD_EXAMPLES=OFF \
        -DBUILD_JAVA=OFF \
        -DBUILD_opencv_python_tests=OFF \
        -DCMAKE_INSTALL_PREFIX=/usr/local \
        -DCMAKE_INSTALL_LIBDIR=lib \
        -DINSTALL_PKGCONFIG=ON \
        -DOPENCV_GENERATE_PKGCONFIG=ON \
        -DPKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
        -DINSTALL_PYTHON_EXAMPLES=OFF \
        -DINSTALL_C_EXAMPLES=OFF && \
    nice -n 19 ionice -c3 make -j1 && \
    make install && \
    cd / && \
    rm -rf /opencv /opencv_contrib

# Install PCL
RUN cd / && \
    git clone --depth 1 --branch pcl-${PCL_VERSION_MAJOR}.${PCL_VERSION_MINOR}.${PCL_VERSION_PATCH} https://github.com/PointCloudLibrary/pcl && \
    mkdir -p /pcl/build && \
    cd /pcl/build && \
    cmake .. \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_apps=OFF \
        -DBUILD_GPU=OFF \
        -DBUILD_CUDA=OFF \
        -DBUILD_examples=OFF \
        -DBUILD_global_tests=OFF \
        -DBUILD_simulation=OFF \
        -DCUDA_BUILD_EMULATION=OFF \
        -DCMAKE_CXX_FLAGS=-std=c++17 \
        -DPCL_ENABLE_SSE=ON \
        -DPCL_SHARED_LIBS=ON \
        -DWITH_VTK=OFF \
        -DPCL_ONLY_CORE_POINT_TYPES=ON \
        -DPCL_COMMON_WARNINGS=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pcl

# Install Pybind11
RUN cd / && \
    git clone --depth 1 --branch v${PYBIND11_VERSION_MAJOR}.${PYBIND11_VERSION_MINOR}.${PYBIND11_VERSION_PATCH} https://github.com/pybind/pybind11 && \
    mkdir -p /pybind11/build && \
    cd /pybind11/build && \
    cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /pybind11

# Install YAML-CPP
RUN cd / && \
    git clone --depth 1 --branch ${YAML_CPP_VERSION_MAJOR}.${YAML_CPP_VERSION_MINOR}.${YAML_CPP_VERSION_PATCH} https://github.com/jbeder/yaml-cpp && \
    mkdir -p /yaml-cpp/build && \
    cd /yaml-cpp/build && \
    cmake .. \
        -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
        -DBUILD_TESTING=OFF \
        -DCMAKE_BUILD_TYPE=Release \
        -DINSTALL_GTEST=OFF \
        -DYAML_CPP_BUILD_TESTS=OFF \
        -DYAML_BUILD_SHARED_LIBS=ON && \
    make -j$(nproc) && \
    make install && \
    cd / && \
    rm -rf /yaml-cpp

SHELL ["/bin/bash", "--login", "-c"]

# Install Python packages directly with system Python to add additional conda env since docker should provide isolated environment
RUN apt-get remove -y python3-blinker || true && \
    pip3 install --upgrade pip setuptools wheel && \
    pip3 install torch==2.6.0 torchvision==0.21.0 torchaudio && \
    pip3 install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu124.html && \
    pip3 install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@stable" && \
    pip3 install --break-system-packages --force-reinstall blinker && \
    pip3 install trimesh wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo pyrender jupyterlab ninja "Cython>=0.29.37" yacs && \
    pip3 install scipy scikit-learn


ENV CUDA_HOME /usr/local/cuda
ENV LD_LIBRARY_PATH /usr/local/cuda/lib64
ENV OPENCV_IO_ENABLE_OPENEXR=1
# Fix multiprocessing issues
ENV PYTHONUNBUFFERED=1
ENV OMP_NUM_THREADS=1


RUN imageio_download_bin freeimage
#### Kaolin will change numpy version
RUN pip3 install --break-system-packages --force-reinstall blinker && \
    pip3 install numpy==1.26.4 transformations einops scikit-image awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui pytinyrenderer PyQt5 cython-npm chardet openpyxl

RUN apt-get update --fix-missing && \
    apt install -y rsync lbzip2 pigz zip p7zip-full p7zip-rar

# ZED SDK + Python API (pyzed) for scripts/data_record_bundlesdf.py.
# Use SDK 4.2 (not 5.0): pyzed 5.0 forces numpy>=2.0, which breaks the
# numpy==1.26.4 pin that kaolin/pytorch3d/my_cpp need. pyzed 4.2 keeps numpy 1.x.
# skip_drivers: no ZED-X kernel modules (USB ZED only); skip_cuda: CUDA already present.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends lsb-release wget less udev sudo zstd libusb-1.0-0 && \
    rm -rf /var/lib/apt/lists/* && \
    cd /tmp && \
    wget -q -O ZED_SDK.run https://download.stereolabs.com/zedsdk/4.2/cu12/ubuntu22 && \
    chmod +x ZED_SDK.run && \
    ./ZED_SDK.run -- silent skip_tools skip_drivers skip_cuda && \
    rm -f ZED_SDK.run && \
    python3 -m pip install --break-system-packages cython && \
    python3 /usr/local/zed/get_python_api.py && \
    pip3 install --break-system-packages --ignore-installed --no-deps "numpy==1.26.4"
# Guard the numpy pin: torch 2.6 (cu124) needs numpy 1.26.4 for torch.from_numpy
# to work (a stray numpy 2.x leaves from_numpy broken: "expected np.ndarray").
# pyzed 4.2 keeps numpy 1.x, so this just re-asserts the pin. --ignore-installed
# overwrites the files in place WITHOUT an uninstall step (uninstalling a
# get_python_api-installed numpy can fail with "uninstall-no-record-file").

# SAM2 real-time fork for scripts/make_masks_sam2.py (single-object mask gen).
# Same fork FoundationPoseROS2 uses; make_masks_sam2.py uses its video predictor
# (build_sam2_video_predictor) for multi-frame interactive labeling.
# --no-deps: fork declares torch>=2.3.1 but we already have torch 2.6 + numpy
# 1.26.4 — don't let it pull deps and disturb the numpy pin.
# hydra-core + iopath are SAM2 runtime deps (config loading); install them
# explicitly since -e . runs with --no-deps. Neither pulls numpy.
# TORCH_CUDA_ARCH_LIST=8.9 + FORCE_CUDA=1: the fork's setup.py unconditionally
# builds the sam2._C CUDA ext, and docker build has no GPU — without a fixed arch
# list torch probes the (absent) device and dies with "IndexError: list index
# out of range". 8.9 = RTX 4090, matching OpenCV's CUDA_ARCH_BIN above.
RUN git clone --depth 1 https://github.com/Gy920/segment-anything-2-real-time /opt/sam2 && \
    cd /opt/sam2 && \
    TORCH_CUDA_ARCH_LIST="8.9" FORCE_CUDA=1 \
        pip3 install --break-system-packages --no-build-isolation --no-deps -e . && \
    pip3 install --break-system-packages hydra-core iopath && \
    mkdir -p /opt/sam2/checkpoints
# Tk for make_masks_sam2.py's interactive labeler GUI (Pillow already present).
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends python3-tk tk && \
    rm -rf /var/lib/apt/lists/*
# Checkpoint (sam2.1_hiera_small.pt, ~185MB) is NOT baked — bind-mounted from
# .docker_assets/sam2_checkpoints/ (see docker-compose.yaml). Populate once:
#   mkdir -p .docker_assets/sam2_checkpoints && \
#   curl -fL https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt \
#        -o .docker_assets/sam2_checkpoints/sam2.1_hiera_small.pt

# Land in the live-mounted repo by default (base image WORKDIR is the deepstream dir).
# The dir is created empty at build time and the docker-compose volume mounts over it.
WORKDIR /workspace/BundleSDF

# Wire the project entrypoint + env (build context is the repo root, see compose).
#  - entrypoint: sources env, builds mycuda+BundleTrack on first start (if missing),
#    then chains the base DeepStream entrypoint.
#  - /root/.bashrc sources the env too, so interactive `docker exec ... bash` shells
#    get CUDA_HOME/PYTHONPATH/torch lib paths without going through the entrypoint.
COPY docker/bundlesdf.env /etc/bundlesdf.env
COPY docker/entrypoint.sh /usr/local/bin/bundlesdf-entrypoint.sh
RUN chmod +x /usr/local/bin/bundlesdf-entrypoint.sh && \
    echo '[ -f /etc/bundlesdf.env ] && source /etc/bundlesdf.env' >> /root/.bashrc
ENTRYPOINT ["/usr/local/bin/bundlesdf-entrypoint.sh"]
