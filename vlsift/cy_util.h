#include <Python.h>
#include "sift/vl/generic.h"


void set_python_vl_printf() {
    vl_set_printf_func((printf_func_t)PySys_WriteStdout);
}
