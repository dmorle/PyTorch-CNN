#define PY_SSIZE_T_CLEAN
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

#include <stdio.h>
#include <stdlib.h>
#include <Python.h>
#include <byteswap.h>
#include <numpy/arrayobject.h>

#define BUFITEM_READ_SIZE 1024

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
        "MnsitIO",   /* name of module */
        NULL, /* module documentation, may be NULL */
        -1,       /* size of per-interpreter state of the module,
                 or -1 if the module keeps state in global variables. */
        MnistIOMethods
};

// Module name must be test in compile and linked
PyMODINIT_FUNC PyInit_MnistIO(void)
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
 * Helper Functions & structs
 *
 */

// header structure for image files
typedef struct tagImgHeader
{
    unsigned int magicNum;
    unsigned int imgNum;
    unsigned int rowNum;
    unsigned int colNum;
} imgHeader;

// return structure for loading images
typedef struct tagImgData
{
    unsigned char *data;
    int rowNum;
    int colNum;
    int imgNum;
} imgData;

// header structure for label files
typedef struct tagLblHeader
{
    unsigned int magicNum;
    unsigned int lblNum;
} lblHeader;

// TODO: create the necessary helper functions
imgData *loadImageFile(unsigned int magicNum, const char *path)
{
    FILE *pF;
    if ( !(pF = fopen(path, "rb")) )
        return NULL;    // error opening file

    imgHeader *pFileInfo = alloca(sizeof(imgHeader));
    if (fread(pFileInfo, sizeof(imgHeader), 1, pF) != 1)
        return NULL;    // error with file on disk

#if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__

    pFileInfo->magicNum = __bswap_32(pFileInfo->magicNum);
    pFileInfo->imgNum   = __bswap_32(pFileInfo->imgNum);
    pFileInfo->rowNum   = __bswap_32(pFileInfo->rowNum);
    pFileInfo->colNum   = __bswap_32(pFileInfo->colNum);

#endif

    if (pFileInfo->magicNum != magicNum)
        return NULL;    // error with file on disk

    unsigned char *data = (unsigned char *)malloc(
            sizeof(unsigned char) * pFileInfo->imgNum * pFileInfo->colNum * pFileInfo->rowNum);

    imgData* pRet = (imgData *)malloc(sizeof(imgData));

    // while loop reading the binary data
    while (fread(data, BUFITEM_READ_SIZE, 1, pF) == BUFITEM_READ_SIZE);
    fclose(pF);

    pRet->data   = data;
    pRet->rowNum = (int)(pFileInfo->rowNum % INT_MAX);
    pRet->colNum = (int)(pFileInfo->colNum % INT_MAX);
    pRet->imgNum = (int)(pFileInfo->imgNum % INT_MAX);

    return pRet;
}

/*
 *
 * Method definitions
 *
 */

static PyObject *loadTrainingSet(PyObject* self, PyObject* args)
{
    imgData *pRet = loadImageFile(0x00000803, "./data/train-images.idx3-ubyte");
    if (!pRet) {
        fprintf(stderr, "error loading training image file\n");
        Py_RETURN_NONE;
    }

    npy_intp dims[] = {pRet->imgNum, pRet->colNum, pRet->rowNum};

    PyObject *npArr = PyArray_SimpleNewFromData(3, dims, NPY_UBYTE, pRet->data);
    Py_INCREF(npArr);

    free(pRet);
    return npArr;
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