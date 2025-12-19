%module modp
%{
    #include <stdlib.h>
    #include "modp.h"
%}

%typemap(in) int **values {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "[Type error] Input data must be a Python List\n");
        SWIG_fail;
    }

    int m = (int)PyList_Size($input);
    $1 = (int **)malloc(m * sizeof(int *));
    if (!$1) {
        PyErr_SetString(PyExc_TypeError, "[Memory error] Run out of memory\n");
        SWIG_fail;
    }

    for (int i = 0; i < m; i++) {
        PyObject *o = PyList_GetItem($input, i);
        if (!PyList_Check(o)) {
            PyErr_SetString(PyExc_TypeError, "[Type error] Input data must be a Python List\n");
            SWIG_fail;
        }

        int n = (int)PyList_Size(o);

        $1[i] = (int *)malloc(n * sizeof(int));
        if (!$1[i]) {
            PyErr_SetString(PyExc_TypeError, "[Memory error] Run out of memory\n");
            SWIG_fail;
        }

        for (int j = 0; j < n; j++)
            $1[i][j] = (int)PyLong_AsLong(PyList_GetItem(o, j));
    }
}

%typemap(in) int *weights {
    if (!PyList_Check($input)) {
        PyErr_SetString(PyExc_TypeError, "Input data must be a Python List\n");
        SWIG_fail;
    }

    int m = (int)PyList_Size($input);
    $1 = (int *)malloc(m * sizeof(int));

    if (!$1) {
        PyErr_SetString(PyExc_TypeError, "[Memory error] Run out of memory\n");
        SWIG_fail;
    }
    for (int i = 0; i < m; i++)
        $1[i] = (int)PyLong_AsLong(PyList_GetItem($input, i));
}

%typemap(freearg) int **values {
    if ($1) {
        for (int i = 0; i < arg4; i++) {
            if ($1[i])
                free($1[i]);
            $1[i] = NULL;
        }
        free($1);
        $1 = NULL;
    }
}

%typemap(freearg) int *weights {
    if ($1) {
        free($1);
        $1 = NULL;
    }
}

%typemap(out) NDContainer {
    $result = PyList_New($1.size);
    for (int i = 0; i < $1.size; i++) {
        PyObject *o = PyTuple_New($1.dim);
        for (int j = 0; j < $1.dim; j++) {
            PyTuple_SetItem(o, j, PyLong_FromLong($1.ndPoints[i][j]));
        }
        PyList_SetItem($result, i, o);
    }

    for (int i = 0; i < $1.size; i++) {
        free($1.ndPoints[i]);
    }
    free($1.ndPoints);
}

%include "modp.h"
