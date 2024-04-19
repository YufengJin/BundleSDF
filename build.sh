ROOT=$(pwd)
cd /kaolin && pip install -e .
cd ${ROOT}/mycuda && rm -rf build *egg* && pip install -e .
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && cmake .. && make -j11
cd ${ROOT}/submodules && pip install -e submodules/diff-gaussian-rasterization-w-depth
cd ${ROOT}/submodules && pip install -e submodules/simple-knn                      
cd ${ROOT} && pip install -r requirements.txt
