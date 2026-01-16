#pragma once
#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H

#include "base.h" // GPUMatrix klasa

namespace Householder {

    // Kernel 1: Update Pivot Element
    // Racuna normu pivot reda i azurira pivot element.
    __global__ void updatePivotKernel(float* matrix, int pivotRow, int numCols, int ld);

    // Kernel 2: Update Matrix
    // Azurira ostale redove na osnovu pivot reda.
    __global__ void updateMatrixKernel(float* matrix, int pivotRow, int numRows, int numCols, int ld, int startRow);

    // Glavna Host funkcija koja implementira Stream logiku
    void qr_decomposition(GPUMatrix& A);

}

#endif // HOUSEHOLDER_H