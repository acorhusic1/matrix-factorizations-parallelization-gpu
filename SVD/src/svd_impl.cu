// svd_impl.cu — Bidiagonal D&C SVD with GPU R-diagonalization
#include "svd.cuh"
#include "gebrd.cuh"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <algorithm>
#include <numeric>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cusolverDn.h>

#define CHECK_CUDA(call) do {                                                  \
    cudaError_t err = (call);                                                  \
    if (err != cudaSuccess) {                                                  \
        fprintf(stderr, "CUDA error %s:%d: %s\n", __FILE__, __LINE__,         \
                cudaGetErrorString(err)); exit(1); }                           \
} while (0)

#define CHECK_CUBLAS(call) do {                                                \
    cublasStatus_t st = (call);                                                \
    if (st != CUBLAS_STATUS_SUCCESS) {                                         \
        fprintf(stderr, "cuBLAS error %s:%d: %d\n", __FILE__, __LINE__,       \
                (int)st); exit(1); }                                           \
} while (0)

#define CHECK_CUSOLVER(call) do {                                              \
    cusolverStatus_t st = (call);                                              \
    if (st != CUSOLVER_STATUS_SUCCESS) {                                       \
        fprintf(stderr, "cuSOLVER error %s:%d: %d\n", __FILE__, __LINE__,     \
                (int)st); exit(1); }                                           \
} while (0)

static constexpr double EPS = 2.220446049250313e-16;
static constexpr int    LEAF_SIZE = 64;
static constexpr int    GPU_MERGE_THRESH = 64;

#define DEV_EPS 2.220446049250313e-16

// Shared-memory secular solve
__global__ void secular_solve_kernel(
    int na,
    const double* __restrict__ Da,
    const double* __restrict__ za,
    double rho, double rhoinv,
    double* __restrict__ lam_new,
    double* __restrict__ delta_mat,
    int use_smem)
{
    extern __shared__ double smem[];
    const double* s_Da;
    const double* s_za;

    if (use_smem) {
        double* sm_Da = smem;
        double* sm_za = smem + na;
        for (int j = threadIdx.x; j < na; j += blockDim.x) {
            sm_Da[j] = Da[j];
            sm_za[j] = za[j];
        }
        __syncthreads();
        s_Da = sm_Da;
        s_za = sm_za;
    } else {
        s_Da = Da;
        s_za = za;
    }

    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;

    bool last_root = (i == na - 1);
    double origin = s_Da[i];

    double bracket_lo = 0.0;
    double bracket_hi;
    if (last_root) {
        double zn2 = 0;
        for (int j = 0; j < na; j++) zn2 += s_za[j] * s_za[j];
        bracket_hi = rho * zn2;
    } else {
        bracket_hi = s_Da[i + 1] - s_Da[i];
    }

    double tau = 0.5 * bracket_hi;

    for (int it = 0; it < 300; it++) {
        double f = rhoinv;
        double df = 0.0;
        for (int j = 0; j < na; j++) {
            double delta_j = (s_Da[j] - origin) - tau;
            delta_j += copysign(1e-300, delta_j);
            double t = s_za[j] * s_za[j] / delta_j;
            f += t;
            df += t / delta_j;
        }

        if (f < 0) bracket_lo = tau; else bracket_hi = tau;
        if (bracket_hi - bracket_lo < DEV_EPS * 4.0 * fabs(origin + tau) + 1e-300) break;
        if (fabs(f) < DEV_EPS * 10.0) break;

        double eta = -f / df;
        double tau_new = tau + eta;
        if (tau_new <= bracket_lo || tau_new >= bracket_hi)
            tau_new = 0.5 * (bracket_lo + bracket_hi);
        if (fabs(tau_new - tau) < DEV_EPS * 2.0 * (fabs(tau) + fabs(origin) + 1e-300)) {
            tau = tau_new;
            break;
        }
        tau = tau_new;
    }
    lam_new[i] = origin + tau;

    for (int j = 0; j < na; j++)
        delta_mat[(size_t)i * na + j] = (origin - s_Da[j]) + tau;
}

// z_tilde (Gu-Eisenstat product formula)
__global__ void ztilde_kernel(
    int na,
    const double* __restrict__ Da,
    const double* __restrict__ za,
    const double* __restrict__ delta_mat,
    double rho,
    double* __restrict__ z_tilde)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;

    double wi = delta_mat[(size_t)i * na + i];
    for (int j = 0; j < na; j++) {
        if (j == i) continue;
        double num = delta_mat[(size_t)j * na + i];
        double den = Da[i] - Da[j];
        den += copysign(1e-300, den);
        wi *= num / den;
    }
    z_tilde[i] = copysign(sqrt(fabs(wi / rho)), za[i]);
}

// Eigenvector columns of Vsec
__global__ void eigvec_kernel(
    int na,
    const double* __restrict__ z_tilde,
    const double* __restrict__ delta_mat,
    double* __restrict__ Vsec)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= na) return;

    double nrm = 0;
    for (int j = 0; j < na; j++) {
        double den = -delta_mat[(size_t)i * na + j];
        den += copysign(1e-300, den);
        double v = z_tilde[j] / den;
        Vsec[j + (size_t)i * na] = v;
        nrm += v * v;
    }
    nrm = rsqrt(nrm);
    for (int j = 0; j < na; j++)
        Vsec[j + (size_t)i * na] *= nrm;
}

// Set diagonal matrix: M[i,j] = (i==j) ? diag[i] : 0
__global__ void set_diag_matrix_kernel(int n, const double* __restrict__ diag,
                                        double* __restrict__ M, int ldm)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int r = idx % n;
    int c = idx / n;
    M[r + (size_t)c * ldm] = (r == c) ? diag[r] : 0.0;
}

// MV kernel: MV[:, col] = diag(Da) * Vsec[:, col] + rho_signed * za * (za^T * Vsec[:, col])
__global__ void mv_kernel(
    int na,
    const double* __restrict__ Da,
    const double* __restrict__ za,
    const double* __restrict__ Vsec,
    double rho_signed,
    double* __restrict__ MV)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= na) return;

    double zv = 0;
    for (int r = 0; r < na; r++)
        zv += za[r] * Vsec[r + (size_t)col * na];

    for (int r = 0; r < na; r++)
        MV[r + (size_t)col * na] = Da[r] * Vsec[r + (size_t)col * na] + rho_signed * za[r] * zv;
}

// Set matrix to identity
__global__ void set_identity_kernel(int n, double* __restrict__ A, int lda)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int r = idx % n;
    int c = idx / n;
    A[r + (size_t)c * lda] = (r == c) ? 1.0 : 0.0;
}

// Givens rotation on two columns
__global__ void givens_cols_kernel(int n, double* __restrict__ col_i, double* __restrict__ col_j,
                                    double c, double s)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= n) return;
    double a = col_i[r], b = col_j[r];
    col_i[r] =  c*a + s*b;
    col_j[r] = -s*a + c*b;
}

// Un-permute + sort
__global__ void unperm_sort_kernel(int n,
    const double* __restrict__ Qm, int ldqm,
    double* __restrict__ Vout, int ldv,
    const int* __restrict__ perm,
    const int* __restrict__ sp)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int i = idx % n;
    int col = idx / n;
    Vout[perm[i] + (size_t)col * ldv] = Qm[i + (size_t)sp[col] * ldqm];
}

// Extract rows: dst[r, col] = src[r + row_offset, col]
__global__ void extract_rows_kernel(int nrows, int ncols, int row_offset,
    const double* __restrict__ src, int lds,
    double* __restrict__ dst, int ldd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * ncols;
    if (idx >= total) return;
    int r = idx % nrows;
    int col = idx / nrows;
    dst[r + (size_t)col * ldd] = src[(r + row_offset) + (size_t)col * lds];
}

// Write rows back
__global__ void scatter_rows_kernel(int nrows, int ncols, int row_offset,
    const double* __restrict__ src, int lds,
    double* __restrict__ dst, int ldd)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nrows * ncols;
    if (idx >= total) return;
    int r = idx % nrows;
    int col = idx / nrows;
    dst[(r + row_offset) + (size_t)col * ldd] = src[r + (size_t)col * lds];
}

// Gather active columns
__global__ void gather_cols_kernel(int n, int na,
    const double* __restrict__ Qd, int ldq,
    double* __restrict__ Qd_act, int lda,
    const int* __restrict__ d_aidx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * na;
    if (idx >= total) return;
    int r = idx % n;
    int jj = idx / n;
    Qd_act[r + (size_t)jj * lda] = Qd[r + (size_t)d_aidx[jj] * ldq];
}

// Scatter active columns
__global__ void scatter_cols_kernel(int n, int na,
    const double* __restrict__ Qm_act, int lda,
    double* __restrict__ Qm, int ldq,
    const int* __restrict__ d_aidx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * na;
    if (idx >= total) return;
    int r = idx % n;
    int ii = idx / n;
    Qm[r + (size_t)d_aidx[ii] * ldq] = Qm_act[r + (size_t)ii * lda];
}

// Copy deflated columns
__global__ void copy_defl_cols_kernel(int n, int n_defl,
    const double* __restrict__ src, int lds,
    double* __restrict__ dst, int ldd,
    const int* __restrict__ d_defl_idx)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n_defl;
    if (idx >= total) return;
    int r = idx % n;
    int di_idx = idx / n;
    int col = d_defl_idx[di_idx];
    dst[r + (size_t)col * ldd] = src[r + (size_t)col * lds];
}

// Build VbT from V_eig with permutation
__global__ void build_VbT_kernel(int n, int ldvt,
    const double* __restrict__ V_eig,
    const int* __restrict__ sp,
    double* __restrict__ VbT)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int i = idx % n;     // row in VbT
    int j = idx / n;     // col in VbT
    VbT[i + (size_t)j * ldvt] = V_eig[j + (size_t)sp[i] * n];
}

// Build Ub = B * V * Sigma^{-1}
__global__ void build_Ub_kernel(int n, int ldu, int ldvt,
    const double* __restrict__ d_orig,
    const double* __restrict__ e_orig,
    const double* __restrict__ VbT,
    const double* __restrict__ sigma,
    double* __restrict__ Ub)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * n;
    if (idx >= total) return;
    int i  = idx % n;   // row
    int sv = idx / n;    // column (which singular value)

    double sig = sigma[sv];
    if (sig < 1e-300) {
        Ub[i + (size_t)sv * ldu] = (i == sv) ? 1.0 : 0.0;
        return;
    }

    double vi  = VbT[sv + (size_t)i * ldvt];
    double bv  = d_orig[i] * vi;
    if (i < n - 1) {
        bv += e_orig[i] * VbT[sv + (size_t)(i + 1) * ldvt];
    }
    Ub[i + (size_t)sv * ldu] = bv / sig;
}

// Givens rotation (host)
static inline void drot_cols(int m, double* col1, double* col2, double c, double s)
{
    for (int i = 0; i < m; i++) {
        double a = col1[i], b = col2[i];
        col1[i] =  c*a + s*b;
        col2[i] = -s*a + c*b;
    }
}

// Tridiagonal eigensolver — Jacobi (leaf solver)
static void tridiag_eigen(int n, const double* diag_in, const double* offdiag_in,
                          double* eigvals, double* eigvecs, int ldv)
{
    if (n <= 0) return;
    if (n == 1) {
        eigvals[0] = diag_in[0];
        eigvecs[0] = 1.0;
        return;
    }

    std::vector<double> A((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++) A[i + (size_t)i * n] = diag_in[i];
    for (int i = 0; i < n - 1; i++) {
        A[i + (size_t)(i+1) * n] = offdiag_in[i];
        A[(i+1) + (size_t)i * n] = offdiag_in[i];
    }

    memset(eigvecs, 0, (size_t)n * ldv * sizeof(double));
    for (int i = 0; i < n; i++) eigvecs[i + (size_t)i * ldv] = 1.0;

    const int MAX_SWEEPS = 100;
    for (int sweep = 0; sweep < MAX_SWEEPS; sweep++) {
        double off = 0.0;
        for (int p = 0; p < n; p++)
            for (int q = p + 1; q < n; q++)
                off += A[p + (size_t)q * n] * A[p + (size_t)q * n];
        if (sqrt(off) < EPS * n) break;

        for (int p = 0; p < n - 1; p++) {
            for (int q = p + 1; q < n; q++) {
                double apq = A[p + (size_t)q * n];
                if (fabs(apq) < EPS * sqrt(fabs(A[p + (size_t)p * n]) *
                                            fabs(A[q + (size_t)q * n]) + 1e-300))
                    continue;

                double tau = (A[q + (size_t)q * n] - A[p + (size_t)p * n]) / (2.0 * apq);
                double t;
                if (tau >= 0)
                    t = 1.0 / (tau + sqrt(1.0 + tau * tau));
                else
                    t = -1.0 / (-tau + sqrt(1.0 + tau * tau));
                double c = 1.0 / sqrt(1.0 + t * t);
                double s = t * c;

                double app = A[p + (size_t)p * n], aqq = A[q + (size_t)q * n];
                A[p + (size_t)p * n] = app - t * apq;
                A[q + (size_t)q * n] = aqq + t * apq;
                A[p + (size_t)q * n] = 0.0;
                A[q + (size_t)p * n] = 0.0;

                for (int r = 0; r < n; r++) {
                    if (r == p || r == q) continue;
                    double arp = A[r + (size_t)p * n], arq = A[r + (size_t)q * n];
                    A[r + (size_t)p * n] = c * arp - s * arq;
                    A[p + (size_t)r * n] = c * arp - s * arq;
                    A[r + (size_t)q * n] = s * arp + c * arq;
                    A[q + (size_t)r * n] = s * arp + c * arq;
                }

                for (int r = 0; r < n; r++) {
                    double vp = eigvecs[r + (size_t)p * ldv];
                    double vq = eigvecs[r + (size_t)q * ldv];
                    eigvecs[r + (size_t)p * ldv] = c * vp - s * vq;
                    eigvecs[r + (size_t)q * ldv] = s * vp + c * vq;
                }
            }
        }
    }

    for (int i = 0; i < n; i++) eigvals[i] = A[i + (size_t)i * n];
}

// Device memory pool (bump allocator)
struct DevPool {
    double* base;
    size_t  cap;
    size_t  used;

    double* alloc(size_t count) {
        size_t aligned = (used + 15) & ~(size_t)15;
        if (aligned + count > cap) {
            fprintf(stderr, "DevPool OOM: need %zu + %zu, cap %zu\n", aligned, count, cap);
            exit(1);
        }
        double* ptr = base + aligned;
        used = aligned + count;
        return ptr;
    }

    size_t save() const { return used; }
    void restore(size_t mark) { used = mark; }
};

// Cuppen's D&C for symmetric tridiagonal eigenproblem
static void tridiag_dc(int n, const double* diag_in, const double* offdiag_in,
                       double* eigvals,
                       double* d_Q_out, int ldq,
                       cublasHandle_t cublas_h, cusolverDnHandle_t cusolver_h,
                       DevPool& pool)
{
    if (n <= 0) return;
    if (n == 1) {
        eigvals[0] = diag_in[0];
        double one = 1.0;
        CHECK_CUDA(cudaMemcpy(d_Q_out, &one, sizeof(double), cudaMemcpyHostToDevice));
        return;
    }

    if (n <= LEAF_SIZE) {
        std::vector<double> h_eigvecs((size_t)n * n);
        tridiag_eigen(n, diag_in, offdiag_in, eigvals, h_eigvecs.data(), n);
        CHECK_CUDA(cudaMemcpy(d_Q_out, h_eigvecs.data(), (size_t)n * n * sizeof(double), cudaMemcpyHostToDevice));
        return;
    }

    int k = n / 2, n1 = k, n2 = n - k;
    double beta = offdiag_in[k - 1];
    double rho  = fabs(beta);

    std::vector<double> d1(diag_in, diag_in + n1);
    std::vector<double> e1(offdiag_in, offdiag_in + n1 - 1);
    d1[n1 - 1] -= beta;

    std::vector<double> d2(diag_in + k, diag_in + n);
    std::vector<double> e2(offdiag_in + k, offdiag_in + n - 1);
    d2[0] -= beta;

    // Allocate Q1, Q2 on device from pool
    size_t mark_children = pool.save();
    double* d_Q1 = pool.alloc((size_t)n1 * n1);
    double* d_Q2 = pool.alloc((size_t)n2 * n2);

    // Recurse
    std::vector<double> lam1(n1), lam2(n2);
    tridiag_dc(n1, d1.data(), e1.data(), lam1.data(), d_Q1, n1, cublas_h, cusolver_h, pool);
    tridiag_dc(n2, d2.data(), e2.data(), lam2.data(), d_Q2, n2, cublas_h, cusolver_h, pool);

    // ------- Extract last row of Q1, first row of Q2 via kernel -------
    std::vector<double> q1_lastrow(n1), q2_firstrow(n2);
    {
        size_t row_mark = pool.save();
        double* d_row1 = pool.alloc(n1);
        extract_rows_kernel<<<(n1+255)/256, 256>>>(1, n1, n1-1, d_Q1, n1, d_row1, 1);
        CHECK_CUDA(cudaMemcpy(q1_lastrow.data(), d_row1, n1*sizeof(double), cudaMemcpyDeviceToHost));
        double* d_row2 = pool.alloc(n2);
        extract_rows_kernel<<<(n2+255)/256, 256>>>(1, n2, 0, d_Q2, n2, d_row2, 1);
        CHECK_CUDA(cudaMemcpy(q2_firstrow.data(), d_row2, n2*sizeof(double), cudaMemcpyDeviceToHost));
        pool.restore(row_mark);
    }

    // ------- Build rank-1 merge problem on host -------
    std::vector<double> D(n), z(n);
    for (int j = 0; j < n1; j++) { D[j] = lam1[j]; z[j] = q1_lastrow[j]; }
    for (int j = 0; j < n2; j++) { D[n1+j] = lam2[j]; z[n1+j] = q2_firstrow[j]; }

    bool negated = (beta < 0);
    if (negated)
        for (int j = 0; j < n; j++) D[j] = -D[j];

    // Sort D
    std::vector<int> perm(n);
    std::iota(perm.begin(), perm.end(), 0);
    std::sort(perm.begin(), perm.end(), [&](int a, int b) { return D[a] < D[b]; });

    std::vector<double> Ds(n), zs(n);
    for (int i = 0; i < n; i++) { Ds[i] = D[perm[i]]; zs[i] = z[perm[i]]; }

    // ------- Deflation -------
    double normT = 0;
    for (int i = 0; i < n; i++) normT = std::max(normT, fabs(Ds[i]));
    { double zn = 0; for (int i = 0; i < n; i++) zn += zs[i]*zs[i]; normT = std::max(normT, rho*sqrt(zn)); }
    double tol = 8.0 * EPS * normT;

    struct GivensRot { int i, j; double c, s; };
    std::vector<GivensRot> givens_rots;
    std::vector<bool> active(n, true);

    for (int i = 0; i < n; i++) {
        if (fabs(rho * zs[i]) < tol) { zs[i] = 0.0; active[i] = false; }
    }
    for (int i = 0; i < n; i++) {
        if (!active[i]) continue;
        for (int j = i + 1; j < n; j++) {
            if (!active[j]) continue;
            if (fabs(Ds[i] - Ds[j]) < tol) {
                double r = sqrt(zs[i]*zs[i] + zs[j]*zs[j]);
                double c = zs[j] / r, s = zs[i] / r;
                zs[j] = r; zs[i] = 0.0;
                Ds[i] = Ds[j];
                givens_rots.push_back({i, j, c, s});
                active[i] = false;
                break;
            }
        }
    }

    std::vector<int> aidx, defl_idx;
    for (int i = 0; i < n; i++) {
        if (active[i]) aidx.push_back(i);
        else defl_idx.push_back(i);
    }
    int na = (int)aidx.size();
    int n_defl = (int)defl_idx.size();

    std::vector<double> Da(na), za(na);
    for (int i = 0; i < na; i++) { Da[i] = Ds[aidx[i]]; za[i] = zs[aidx[i]]; }
    double rhoinv = 1.0 / rho;
    std::vector<double> lam_new(na);

    // GPU merge path
    if (na >= GPU_MERGE_THRESH && cublas_h) {
        size_t mark_merge = pool.save();

        // Secular solve + eigenvectors
        double* d_Da    = pool.alloc(na);
        double* d_za    = pool.alloc(na);
        double* d_lam   = pool.alloc(na);
        double* d_zt    = pool.alloc(na);
        double* d_delta = pool.alloc((size_t)na * na);
        double* d_Vsec  = pool.alloc((size_t)na * na);

        CHECK_CUDA(cudaMemcpy(d_Da, Da.data(), na*sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_za, za.data(), na*sizeof(double), cudaMemcpyHostToDevice));

        int blk = 256;
        int grd = (na + blk - 1) / blk;
        size_t smem_bytes = 2 * na * sizeof(double);
        int use_smem = 1;
        if (smem_bytes > 48 * 1024) { smem_bytes = 0; use_smem = 0; }

        secular_solve_kernel<<<grd, blk, smem_bytes>>>(na, d_Da, d_za, rho, rhoinv, d_lam, d_delta, use_smem);
        ztilde_kernel<<<grd, blk>>>(na, d_Da, d_za, d_delta, rho, d_zt);
        eigvec_kernel<<<grd, blk>>>(na, d_zt, d_delta, d_Vsec);

        CHECK_CUDA(cudaMemcpy(lam_new.data(), d_lam, na*sizeof(double), cudaMemcpyDeviceToHost));

        // R-diagonalization: R = V^T*M*V, diagonalize, apply rotation
        {
            double one = 1.0, zero = 0.0;

            // MV = M * V
            double* d_MV = pool.alloc((size_t)na * na);
            mv_kernel<<<(na + 255) / 256, 256>>>(na, d_Da, d_za, d_Vsec, rho, d_MV);

            // R = V^T * MV
            double* d_R = d_delta;  // reuse na×na delta space
            CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                na, na, na, &one, d_Vsec, na, d_MV, na, &zero, d_R, na));

            // Diagonalize R via dsyevd
            int lwork = 0;
            CHECK_CUSOLVER(cusolverDnDsyevd_bufferSize(cusolver_h,
                CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                na, d_R, na, d_lam, &lwork));


            double* d_sywork = nullptr;
            int* d_info_sy = nullptr;
            CHECK_CUDA(cudaMalloc(&d_sywork, (lwork > 0 ? lwork : 1) * sizeof(double)));
            CHECK_CUDA(cudaMalloc(&d_info_sy, sizeof(int)));

            CHECK_CUSOLVER(cusolverDnDsyevd(cusolver_h,
                CUSOLVER_EIG_MODE_VECTOR, CUBLAS_FILL_MODE_LOWER,
                na, d_R, na, d_lam, d_sywork, lwork, d_info_sy));

            int h_info = 0;
            CHECK_CUDA(cudaMemcpy(&h_info, d_info_sy, sizeof(int), cudaMemcpyDeviceToHost));
            cudaFree(d_sywork);
            cudaFree(d_info_sy);

            if (h_info != 0) {
                fprintf(stderr, "dsyevd failed: info=%d\n", h_info);
            }

            // V_corrected = V * G
            CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                na, na, na, &one, d_Vsec, na, d_R, na, &zero, d_MV, na));


            CHECK_CUDA(cudaMemcpy(d_Vsec, d_MV, (size_t)na * na * sizeof(double), cudaMemcpyDeviceToDevice));


            CHECK_CUDA(cudaMemcpy(lam_new.data(), d_lam, na * sizeof(double), cudaMemcpyDeviceToHost));
        }

        if (negated) {
            for (int i = 0; i < n; i++)  Ds[i] = -Ds[i];
            for (int i = 0; i < na; i++) Da[i] = -Da[i];
            for (int i = 0; i < na; i++) lam_new[i] = -lam_new[i];
        }

        double* d_Vs2 = d_Vsec;

        // Assembly
        double* d_Qd = pool.alloc((size_t)n * n);
        {
            int total = n * n;
            set_identity_kernel<<<(total+255)/256, 256>>>(n, d_Qd, n);
        }
        for (auto& gr : givens_rots) {
            givens_cols_kernel<<<(n+255)/256, 256>>>(n,
                d_Qd + (size_t)gr.i * n,
                d_Qd + (size_t)gr.j * n,
                gr.c, gr.s);
        }

        size_t int_doubles = ((size_t)(na + n_defl) * sizeof(int) + sizeof(double) - 1) / sizeof(double);
        int* d_aidx_arr = (int*)pool.alloc(int_doubles);
        int* d_defl_arr = d_aidx_arr + na;
        CHECK_CUDA(cudaMemcpy(d_aidx_arr, aidx.data(), na*sizeof(int), cudaMemcpyHostToDevice));
        if (n_defl > 0)
            CHECK_CUDA(cudaMemcpy(d_defl_arr, defl_idx.data(), n_defl*sizeof(int), cudaMemcpyHostToDevice));

        double* d_Qm = pool.alloc((size_t)n * n);
        CHECK_CUDA(cudaMemset(d_Qm, 0, (size_t)n*n*sizeof(double)));

        if (n_defl > 0) {
            int total = n * n_defl;
            copy_defl_cols_kernel<<<(total+255)/256, 256>>>(n, n_defl, d_Qd, n, d_Qm, n, d_defl_arr);
        }

        if (na > 0) {
            double* d_Qd_act = pool.alloc((size_t)n * na);
            double* d_Qm_act = pool.alloc((size_t)n * na);
            {
                int total = n * na;
                gather_cols_kernel<<<(total+255)/256, 256>>>(n, na, d_Qd, n, d_Qd_act, n, d_aidx_arr);
            }
            {
                double one = 1.0, zero = 0.0;
                CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                    n, na, na, &one, d_Qd_act, n, d_Vs2, na, &zero, d_Qm_act, n));
            }
            {
                int total = n * na;
                scatter_cols_kernel<<<(total+255)/256, 256>>>(n, na, d_Qm_act, n, d_Qm, n, d_aidx_arr);
            }
        }

        // Sort + un-permute
        std::vector<double> all_lam(n);
        for (int i : defl_idx) all_lam[i] = Ds[i];
        for (int ii = 0; ii < na; ii++) all_lam[aidx[ii]] = lam_new[ii];

        std::vector<int> sp(n);
        std::iota(sp.begin(), sp.end(), 0);
        std::sort(sp.begin(), sp.end(), [&](int a, int b) { return all_lam[a] < all_lam[b]; });

        std::vector<double> lam_sorted(n);
        for (int i = 0; i < n; i++) lam_sorted[i] = all_lam[sp[i]];

        size_t perm_doubles = ((size_t)n * 2 * sizeof(int) + sizeof(double) - 1) / sizeof(double);
        int* d_perm = (int*)pool.alloc(perm_doubles);
        int* d_sp   = d_perm + n;
        CHECK_CUDA(cudaMemcpy(d_perm, perm.data(), n*sizeof(int), cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaMemcpy(d_sp, sp.data(), n*sizeof(int), cudaMemcpyHostToDevice));

        double* d_Vs_final = pool.alloc((size_t)n * n);
        {
            int total = n * n;
            unperm_sort_kernel<<<(total+255)/256, 256>>>(n, d_Qm, n, d_Vs_final, n, d_perm, d_sp);
        }

        // [Q1 0; 0 Q2] * V_sorted
        {
            double one = 1.0, zero = 0.0;
            int nmax = (n1 > n2) ? n1 : n2;
            double* d_temp  = pool.alloc((size_t)nmax * n);
            double* d_Epart = pool.alloc((size_t)nmax * n);

            {
                int total = n1 * n;
                extract_rows_kernel<<<(total+255)/256, 256>>>(n1, n, 0, d_Vs_final, n, d_temp, n1);
            }
            CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                n1, n, n1, &one, d_Q1, n1, d_temp, n1, &zero, d_Epart, n1));
            {
                int total = n1 * n;
                scatter_rows_kernel<<<(total+255)/256, 256>>>(n1, n, 0, d_Epart, n1, d_Q_out, n);
            }

            {
                int total = n2 * n;
                extract_rows_kernel<<<(total+255)/256, 256>>>(n2, n, n1, d_Vs_final, n, d_temp, n2);
            }
            CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
                n2, n, n2, &one, d_Q2, n2, d_temp, n2, &zero, d_Epart, n2));
            {
                int total = n2 * n;
                scatter_rows_kernel<<<(total+255)/256, 256>>>(n2, n, n1, d_Epart, n2, d_Q_out, n);
            }
        }

        pool.restore(mark_children);
        for (int i = 0; i < n; i++) eigvals[i] = lam_sorted[i];
        return;
    }

    // CPU merge path (small na)
    std::vector<double> h_Q1((size_t)n1 * n1), h_Q2((size_t)n2 * n2);
    CHECK_CUDA(cudaMemcpy(h_Q1.data(), d_Q1, (size_t)n1*n1*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_Q2.data(), d_Q2, (size_t)n2*n2*sizeof(double), cudaMemcpyDeviceToHost));

    // Build Qd on host
    std::vector<double> Qd((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++) Qd[i + (size_t)i * n] = 1.0;
    for (auto& gr : givens_rots)
        drot_cols(n, Qd.data() + (size_t)gr.i*n, Qd.data() + (size_t)gr.j*n, gr.c, gr.s);

    if (na > 0) {
        // Secular solve on CPU
        std::vector<double> delta_mat((size_t)na * na);
        for (int i = 0; i < na; i++) {
            bool last_root = (i == na - 1);
            double origin = Da[i];
            double bracket_lo = 0.0, bracket_hi;
            if (last_root) {
                double zn2 = 0;
                for (int j = 0; j < na; j++) zn2 += za[j]*za[j];
                bracket_hi = rho * zn2;
            } else {
                bracket_hi = Da[i + 1] - Da[i];
            }
            double tau = 0.5 * bracket_hi;
            for (int it = 0; it < 300; it++) {
                double f = rhoinv, df = 0.0;
                for (int j = 0; j < na; j++) {
                    double delta_j = (Da[j] - origin) - tau;
                    if (fabs(delta_j) < 1e-300) delta_j = copysign(1e-300, delta_j);
                    double t = za[j]*za[j] / delta_j;
                    f += t; df += t / delta_j;
                }
                if (f < 0) bracket_lo = tau; else bracket_hi = tau;
                if (bracket_hi - bracket_lo < EPS * 4.0 * fabs(origin + tau) + 1e-300) break;
                if (fabs(f) < EPS * 10.0) break;
                double eta = -f / df;
                double tau_new = tau + eta;
                if (tau_new <= bracket_lo || tau_new >= bracket_hi)
                    tau_new = 0.5 * (bracket_lo + bracket_hi);
                if (fabs(tau_new - tau) < EPS * 2.0 * (fabs(tau) + fabs(origin) + 1e-300)) { tau = tau_new; break; }
                tau = tau_new;
            }
            lam_new[i] = origin + tau;
            for (int j = 0; j < na; j++)
                delta_mat[i*(size_t)na + j] = (origin - Da[j]) + tau;
        }

        std::vector<double> z_tilde(na);
        for (int i = 0; i < na; i++) {
            double wi = delta_mat[i*(size_t)na + i];
            for (int j = 0; j < na; j++) {
                if (j == i) continue;
                double num = delta_mat[j*(size_t)na + i];
                double den = Da[i] - Da[j];
                if (fabs(den) < 1e-300) den = copysign(1e-300, den);
                wi *= num / den;
            }
            z_tilde[i] = copysign(sqrt(fabs(wi / rho)), za[i]);
        }

        std::vector<double> Vsec((size_t)na * na);
        for (int i = 0; i < na; i++) {
            double nrm = 0;
            for (int j = 0; j < na; j++) {
                double den = -delta_mat[i*(size_t)na + j];
                if (fabs(den) < 1e-300) den = copysign(1e-300, den);
                double v = z_tilde[j] / den;
                Vsec[j + (size_t)i * na] = v;
                nrm += v*v;
            }
            nrm = sqrt(nrm);
            if (nrm > 0) for (int j = 0; j < na; j++) Vsec[j + (size_t)i*na] /= nrm;
        }

        double rho_signed_cpu = negated ? -rho : rho;
        if (negated) {
            for (int i = 0; i < n; i++)  Ds[i] = -Ds[i];
            for (int i = 0; i < na; i++) Da[i] = -Da[i];
            for (int i = 0; i < na; i++) lam_new[i] = -lam_new[i];
        }

        // Jacobi cleanup
        {
            std::vector<double> R((size_t)na * na, 0.0);
            for (int col = 0; col < na; col++) {
                double zv = 0;
                for (int r = 0; r < na; r++) zv += za[r] * Vsec[r + (size_t)col * na];
                for (int r = 0; r < na; r++) {
                    double mv_r = Da[r] * Vsec[r + (size_t)col * na] + rho_signed_cpu * za[r] * zv;
                    for (int j = 0; j < na; j++)
                        R[j + (size_t)col * na] += Vsec[r + (size_t)j * na] * mv_r;
                }
            }
            for (int sweep = 0; sweep < 3; sweep++) {
                double off = 0;
                for (int p = 0; p < na; p++)
                    for (int q = p+1; q < na; q++)
                        off += R[p + (size_t)q*na] * R[p + (size_t)q*na];
                if (sqrt(off) < EPS * na) break;
                for (int p = 0; p < na - 1; p++) {
                    for (int q = p + 1; q < na; q++) {
                        double rpq = R[p + (size_t)q * na];
                        if (fabs(rpq) < EPS * sqrt(fabs(R[p + (size_t)p*na] * R[q + (size_t)q*na]) + 1e-300))
                            continue;
                        double tau2 = (R[q + (size_t)q*na] - R[p + (size_t)p*na]) / (2.0 * rpq);
                        double t;
                        if (tau2 >= 0) t = 1.0 / (tau2 + sqrt(1.0 + tau2*tau2));
                        else t = -1.0 / (-tau2 + sqrt(1.0 + tau2*tau2));
                        double c = 1.0 / sqrt(1.0 + t*t);
                        double s = t * c;
                        R[p + (size_t)p*na] -= t * rpq;
                        R[q + (size_t)q*na] += t * rpq;
                        R[p + (size_t)q*na] = 0.0;
                        R[q + (size_t)p*na] = 0.0;
                        for (int r = 0; r < na; r++) {
                            if (r == p || r == q) continue;
                            double arp = R[r + (size_t)p*na], arq = R[r + (size_t)q*na];
                            R[r + (size_t)p*na] = c*arp - s*arq;
                            R[p + (size_t)r*na] = c*arp - s*arq;
                            R[r + (size_t)q*na] = s*arp + c*arq;
                            R[q + (size_t)r*na] = s*arp + c*arq;
                        }
                        for (int r = 0; r < na; r++) {
                            double vp = Vsec[r + (size_t)p*na];
                            double vq = Vsec[r + (size_t)q*na];
                            Vsec[r + (size_t)p*na] = c*vp - s*vq;
                            Vsec[r + (size_t)q*na] = s*vp + c*vq;
                        }
                    }
                }
            }
            for (int i = 0; i < na; i++) lam_new[i] = R[i + (size_t)i*na];
        }

        // Assemble eigenvectors
        std::vector<double> Qm((size_t)n * n, 0.0);
        for (int di : defl_idx)
            for (int r = 0; r < n; r++) Qm[r + (size_t)di*n] = Qd[r + (size_t)di*n];

        for (int ii = 0; ii < na; ii++) {
            int col = aidx[ii];
            for (int r = 0; r < n; r++) {
                double s = 0;
                for (int jj = 0; jj < na; jj++)
                    s += Qd[r + (size_t)aidx[jj]*n] * Vsec[jj + (size_t)ii*na];
                Qm[r + (size_t)col*n] = s;
            }
        }
        Qd = std::move(Qm);
    } else {
        // na == 0
        if (negated)
            for (int i = 0; i < n; i++) Ds[i] = -Ds[i];
    }

    // Build eigenvalue array and sort
    std::vector<double> all_lam(n);
    for (int i : defl_idx) all_lam[i] = Ds[i];
    for (int ii = 0; ii < na; ii++) all_lam[aidx[ii]] = lam_new[ii];

    // Un-permute
    std::vector<double> V_unperm((size_t)n * n);
    for (int i = 0; i < n; i++) {
        int orig = perm[i];
        for (int col = 0; col < n; col++)
            V_unperm[orig + (size_t)col*n] = Qd[i + (size_t)col*n];
    }

    std::vector<int> sp(n);
    std::iota(sp.begin(), sp.end(), 0);
    std::sort(sp.begin(), sp.end(), [&](int a, int b) { return all_lam[a] < all_lam[b]; });

    std::vector<double> lam_sorted(n), V_sorted((size_t)n * n);
    for (int i = 0; i < n; i++) {
        lam_sorted[i] = all_lam[sp[i]];
        for (int r = 0; r < n; r++)
            V_sorted[r + (size_t)i*n] = V_unperm[r + (size_t)sp[i]*n];
    }

    // [Q1 0; 0 Q2] * V_sorted
    {
        size_t mark_final = pool.save();
        double* d_V = pool.alloc((size_t)n * n);
        CHECK_CUDA(cudaMemcpy(d_V, V_sorted.data(), (size_t)n*n*sizeof(double), cudaMemcpyHostToDevice));

        double one = 1.0, zero = 0.0;
        int nmax = (n1 > n2) ? n1 : n2;
        double* d_temp  = pool.alloc((size_t)nmax * n);
        double* d_Epart = pool.alloc((size_t)nmax * n);

        // Q1 * V_sorted[0:n1, :]
        {
            int total = n1 * n;
            extract_rows_kernel<<<(total+255)/256, 256>>>(n1, n, 0, d_V, n, d_temp, n1);
        }
        CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            n1, n, n1, &one, d_Q1, n1, d_temp, n1, &zero, d_Epart, n1));
        {
            int total = n1 * n;
            scatter_rows_kernel<<<(total+255)/256, 256>>>(n1, n, 0, d_Epart, n1, d_Q_out, n);
        }

        // Q2 * V_sorted[n1:n, :]
        {
            int total = n2 * n;
            extract_rows_kernel<<<(total+255)/256, 256>>>(n2, n, n1, d_V, n, d_temp, n2);
        }
        CHECK_CUBLAS(cublasDgemm(cublas_h, CUBLAS_OP_N, CUBLAS_OP_N,
            n2, n, n2, &one, d_Q2, n2, d_temp, n2, &zero, d_Epart, n2));
        {
            int total = n2 * n;
            scatter_rows_kernel<<<(total+255)/256, 256>>>(n2, n, n1, d_Epart, n2, d_Q_out, n);
        }

        pool.restore(mark_final);
    }

    pool.restore(mark_children);
    for (int i = 0; i < n; i++) eigvals[i] = lam_sorted[i];
}

// Bidiagonal SVD via tridiagonal D&C
static void bidiag_svd_dc(int n, double* d, double* e,
                           double* Ub, int ldu, double* VbT, int ldvt,
                           cublasHandle_t cublas_h)
{
    if (n <= 0) return;
    if (n == 1) {
        double s = d[0];
        Ub[0] = (s < 0) ? -1.0 : 1.0;
        VbT[0] = 1.0;
        d[0] = fabs(s);
        return;
    }


    size_t pool_doubles = 16 * (size_t)n * n;
    double* d_pool_base = nullptr;
    CHECK_CUDA(cudaMalloc(&d_pool_base, pool_doubles * sizeof(double)));

    DevPool pool;
    pool.base = d_pool_base;
    pool.cap = pool_doubles;
    pool.used = 0;

    std::vector<double> orig_d(d, d + n);
    std::vector<double> orig_e(e, e + n - 1);

    // Form T = B^T*B
    std::vector<double> Td(n), Te(n - 1);
    for (int i = 0; i < n; i++) {
        Td[i] = d[i] * d[i];
        if (i > 0) Td[i] += e[i-1] * e[i-1];
    }
    for (int i = 0; i < n - 1; i++)
        Te[i] = d[i] * e[i];

    double* d_V_eig = pool.alloc((size_t)n * n);

    // Create cuSOLVER handle once for entire D&C recursion
    cusolverDnHandle_t cusolver_h = nullptr;
    CHECK_CUSOLVER(cusolverDnCreate(&cusolver_h));

    std::vector<double> lambda(n);
    tridiag_dc(n, Td.data(), Te.data(), lambda.data(), d_V_eig, n, cublas_h, cusolver_h, pool);
    CHECK_CUDA(cudaDeviceSynchronize());

    // Sort by eigenvalue descending
    std::vector<int> sp(n);
    std::iota(sp.begin(), sp.end(), 0);
    std::sort(sp.begin(), sp.end(), [&](int a, int b) { return lambda[a] > lambda[b]; });

    for (int i = 0; i < n; i++)
        d[i] = sqrt(fabs(lambda[sp[i]]));

    // ---- GPU construction of VbT and Ub ----
    // d_V_eig occupies [0, n*n) in pool
    pool.used = (size_t)n * n;

    double* d_VbT_g = pool.alloc((size_t)n * n);
    double* d_Ub_g  = pool.alloc((size_t)n * n);

    // Upload sp array
    size_t sp_int_doubles = ((size_t)n * sizeof(int) + sizeof(double) - 1) / sizeof(double);
    int* d_sp = (int*)pool.alloc(sp_int_doubles);
    CHECK_CUDA(cudaMemcpy(d_sp, sp.data(), n*sizeof(int), cudaMemcpyHostToDevice));

    // Build VbT on GPU
    {
        int total = n * n;
        build_VbT_kernel<<<(total+255)/256, 256>>>(n, n, d_V_eig, d_sp, d_VbT_g);
    }

    // Upload original d and e for Ub construction
    double* d_orig_d = pool.alloc(n);
    double* d_orig_e = pool.alloc(std::max(1, n-1));
    double* d_sigma  = pool.alloc(n);
    CHECK_CUDA(cudaMemcpy(d_orig_d, orig_d.data(), n*sizeof(double), cudaMemcpyHostToDevice));
    if (n > 1)
        CHECK_CUDA(cudaMemcpy(d_orig_e, orig_e.data(), (n-1)*sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_sigma, d, n*sizeof(double), cudaMemcpyHostToDevice));

    // Build Ub on GPU
    {
        int total = n * n;
        build_Ub_kernel<<<(total+255)/256, 256>>>(n, n, n, d_orig_d, d_orig_e, d_VbT_g, d_sigma, d_Ub_g);
    }

    // Cholesky QR
    {
        double* d_G = pool.alloc((size_t)n * n);

        double one = 1.0, zero = 0.0;
        CHECK_CUBLAS(cublasDsyrk(cublas_h, CUBLAS_FILL_MODE_UPPER, CUBLAS_OP_T,
            n, n, &one, d_Ub_g, n, &zero, d_G, n));

        int lwork = 0;
        CHECK_CUSOLVER(cusolverDnDpotrf_bufferSize(
            cusolver_h, CUBLAS_FILL_MODE_UPPER, n, d_G, n, &lwork));

        double* d_work = pool.alloc(lwork);
        int* d_info = (int*)pool.alloc(1);

        CHECK_CUSOLVER(cusolverDnDpotrf(
            cusolver_h, CUBLAS_FILL_MODE_UPPER, n, d_G, n, d_work, lwork, d_info));

        int h_info = 0;
        CHECK_CUDA(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_info != 0) {
            fprintf(stderr, "CholQR potrf failed: info=%d\n", h_info);
        }

        CHECK_CUBLAS(cublasDtrsm(cublas_h, CUBLAS_SIDE_RIGHT, CUBLAS_FILL_MODE_UPPER,
            CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
            n, n, &one, d_G, n, d_Ub_g, n));
    }

    // Download Ub and VbT from device
    CHECK_CUDA(cudaMemcpy(Ub, d_Ub_g, (size_t)n*n*sizeof(double), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(VbT, d_VbT_g, (size_t)n*n*sizeof(double), cudaMemcpyDeviceToHost));

    cusolverDnDestroy(cusolver_h);
    if (d_pool_base) cudaFree(d_pool_base);
}

// svd_full — A = U * S * VT

size_t svd_workspace_size(int m, int n, int nb)
{
    size_t s = 0;
    s += gebrd_workspace_size(m, n, nb);
    s += orgbr_q_workspace_size(m, n);
    s += orgbr_pt_workspace_size(m, n);
    s += (size_t)m * m;
    s += (size_t)n * n;
    return s;
}

void svd_full(
    cublasHandle_t handle,
    int m, int n, int lda,
    double* A,
    double* S,
    double* U, int ldu,
    double* VT, int ldvt,
    double* work,
    int nb)
{
    int minmn = (m < n) ? m : n;

    double *d_d, *d_e, *d_tauq, *d_taup;
    CHECK_CUDA(cudaMalloc(&d_d,    minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_e,    std::max(1, minmn - 1) * (int)sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_tauq, minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_taup, minmn * sizeof(double)));

    double* gebrd_work = work;

    gebrd_merged_rank2b(handle, m, n, lda, A, d_d, d_e, d_tauq, d_taup, gebrd_work, nb);
    CHECK_CUDA(cudaDeviceSynchronize());

    double* d_Q;
    CHECK_CUDA(cudaMalloc(&d_Q, (size_t)m * m * sizeof(double)));
    size_t ws_q = orgbr_q_workspace_size(m, n);
    double* d_wq;
    CHECK_CUDA(cudaMalloc(&d_wq, ws_q * sizeof(double)));
    orgbr_generate_Q(handle, m, n, lda, A, d_tauq, d_Q, m, d_wq);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_wq);

    double* d_PT;
    CHECK_CUDA(cudaMalloc(&d_PT, (size_t)n * n * sizeof(double)));
    size_t ws_pt = orgbr_pt_workspace_size(m, n);
    double* d_wpt;
    CHECK_CUDA(cudaMalloc(&d_wpt, ws_pt * sizeof(double)));
    orgbr_generate_PT(handle, m, n, lda, A, d_taup, d_PT, n, d_wpt);
    CHECK_CUDA(cudaDeviceSynchronize());
    cudaFree(d_wpt);

    std::vector<double> h_d(minmn), h_e(std::max(1, minmn - 1));
    CHECK_CUDA(cudaMemcpy(h_d.data(), d_d, minmn * sizeof(double), cudaMemcpyDeviceToHost));
    if (minmn > 1)
        CHECK_CUDA(cudaMemcpy(h_e.data(), d_e, (minmn - 1) * sizeof(double), cudaMemcpyDeviceToHost));

    cublasPointerMode_t origMode;
    CHECK_CUBLAS(cublasGetPointerMode(handle, &origMode));
    CHECK_CUBLAS(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

    std::vector<double> h_Ub((size_t)minmn * minmn);
    std::vector<double> h_VbT((size_t)minmn * minmn);

    bidiag_svd_dc(minmn, h_d.data(), h_e.data(),
                  h_Ub.data(), minmn, h_VbT.data(), minmn, handle);

    CHECK_CUDA(cudaMemcpy(S, h_d.data(), minmn * sizeof(double), cudaMemcpyHostToDevice));

    double *d_Ub, *d_VbT;
    CHECK_CUDA(cudaMalloc(&d_Ub,  (size_t)minmn * minmn * sizeof(double)));
    CHECK_CUDA(cudaMalloc(&d_VbT, (size_t)minmn * minmn * sizeof(double)));
    CHECK_CUDA(cudaMemcpy(d_Ub,  h_Ub.data(),  (size_t)minmn * minmn * sizeof(double), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_VbT, h_VbT.data(), (size_t)minmn * minmn * sizeof(double), cudaMemcpyHostToDevice));

    double one = 1.0, zero = 0.0;

    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        m, minmn, minmn, &one, d_Q, m, d_Ub, minmn, &zero, U, ldu));

    CHECK_CUBLAS(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
        minmn, n, minmn, &one, d_VbT, minmn, d_PT, n, &zero, VT, ldvt));

    CHECK_CUBLAS(cublasSetPointerMode(handle, origMode));

    cudaFree(d_d);  cudaFree(d_e);
    cudaFree(d_tauq); cudaFree(d_taup);
    cudaFree(d_Q);  cudaFree(d_PT);
    cudaFree(d_Ub); cudaFree(d_VbT);
}
