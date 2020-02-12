#define PY_SSIZE_T_CLEAN
#include <stdio.h>
#include <Python.h>
#include <numpy/arrayobject.h>

/*
 *
 * Method prototypes
 *
 */

static PyObject *loadTrainingSet(PyObject*, PyObject*);
static PyObject *loadTrainingLabels(PyObject*, PyObject*);
static PyObject *loadTestingSet(PyObject*, PyObject*);
static PyObject *loadTestingLabels(PyObject*, PyObject*);

/*
 *
 * Setup for python module
 *
 */

// Set up the methods table
static PyMethodDef MnistIOMethods[] = {
        {
            "C_loadTrainingSet",
            loadTrainingSet,
            METH_VARARGS,
            "loads the mnist training set from a pre-determined location"
        },
        {
            "C_loadTrainingLabels",
            loadTrainingLabels,
            METH_VARARGS,
            "loads the mnist training labels from a pre-determined location"
        },
        {
            "C_loadTestingSet",
            loadTestingSet,
            METH_VARARGS,
            "loads the mnist testing set from a pre-determined location"
        },
        {
            "C_loadTestingLabels",
            loadTestingLabels,
            METH_VARARGS,
            "loads the mnist testing labels from a pre-determined location"
        },
        {NULL, NULL, 0, NULL}        /* Sentinel */
};

static struct PyModuleDef MnistIOModule = {
        PyModuleDef_HEAD_INIT,
        "demo",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        MnistIOMethods
};

// Module name must be test in compile and linked
PyMODINIT_FUNC PyInit_MnistIO()
{
    import_array()
    return PyModule_Create(&MnistIOModule);
}

int main(int argc, char *argv[])
{
    wchar_t *program = Py_DecodeLocale(argv[0], NULL);
    if (program == NULL) {
        fprintf(stderr, "Fatal error: cannot decode argv[0]\n");
        exit(1);
    }

    /* Add a built-in module, before Py_Initialize */
    PyImport_AppendInittab("MnistIO", PyInit_MnistIO);

    /* Pass argv[0] to the Python interpreter */
    Py_SetProgramName(program);

    /* Initialize the Python interpreter.  Required. */
    Py_Initialize();

    /* Optionally import the module; alternatively,
       import can be deferred until the embedded script
       imports it. */
    PyImport_ImportModule("MnistIO");

    PyMem_RawFree(program);
    return 0;
}

/*
 *
 * Helper Functions
 *
 */

// TODO: create the nessisary helper functions

/*
 *
 * Method definitions
 *
 */

static PyObject *loadTrainingSet(PyObject* self, PyObject* args)
{
    // TODO: load the training set
    Py_RETURN_NONE;
}

static PyObject *loadTrainingLabels(PyObject* self, PyObject* args)
{
    // TODO: load the training labels
    Py_RETURN_NONE;
}

static PyObject *loadTestingSet(PyObject* self, PyObject* args)
{
    // TODO: load the testing set
    Py_RETURN_NONE;
}

static PyObject *loadTestingLabels(PyObject* self, PyObject* args)
{
    // TODO: load the testing labels
    Py_RETURN_NONE;
}