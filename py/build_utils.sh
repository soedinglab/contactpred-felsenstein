NP_INC="$(python -c 'import numpy;print(numpy.get_include())')"
export CFLAGS="$CFLAGS -I$NP_INC -std=c99 -march=native"
cythonize -f -i -a utils.pyx
