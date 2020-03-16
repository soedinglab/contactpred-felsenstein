NP_INC="$(python -c 'import numpy;print(numpy.get_include())')"
export CFLAGS="$CFLAGS -I$NP_INC -std=c99"

cythonize -f -i -a optimize_felsenstein.pyx
