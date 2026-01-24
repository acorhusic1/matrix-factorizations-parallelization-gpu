#pragma once
#ifndef HOUSEHOLDER_H
#define HOUSEHOLDER_H

#include "base.h"
#include <vector>

namespace Householder {

    /*
     * A        Matrix to factorize (in-place).
     *          Output: Upper triangle is R. Strict lower triangle contains v[1:m].
     * h_tau    Host vector to store tau factors (needed for Q extraction).
     */
    void qr_decomposition(GPUMatrix& A, std::vector<float>& h_tau, int block_size = -1);

    // Reconstruct Q from the compressed Householder vectors and Tau factors.
    void extract_Q(const GPUMatrix& A, const std::vector<float>& h_tau, GPUMatrix& Q);

    // Extract R matrix from factorized form.
    void extract_R(const GPUMatrix& A, GPUMatrix& R);

}

#endif // HOUSEHOLDER_H