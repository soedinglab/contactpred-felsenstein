NP_INC="$(python -c 'import numpy;print(numpy.get_include())')"
export CFLAGS="$CFLAGS -I$NP_INC -std=c99 -O3 -DSINGLE_PRECISION"

cythonize -f -i -a optimize_felsenstein_faster_float.pyx
