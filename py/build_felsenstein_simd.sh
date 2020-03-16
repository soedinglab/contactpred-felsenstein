set -e

if [[ -z $FS_REPO ]]; then
  FS_REPO=$PWD/..
fi

FS_SIMD_HEADERS=$FS_REPO/lib/
FS_HEADERS=$FS_REPO

BUILD_DIR=$FS_REPO/build_pyx_deps

NP_INC="$(python -c 'import numpy;print(numpy.get_include())')"
export CFLAGS="$CFLAGS -I$FS_SIMD_HEADERS -I$FS_HEADERS -I$NP_INC -std=c99 -march=native -mavx2 -mfpmath=sse -DAVX2=1"

cd $FS_REPO/py
cythonize -f -i -a pam_distance.pyx
cythonize -f -i -a optimize_felsenstein_simd.pyx
