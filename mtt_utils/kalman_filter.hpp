#pragma once
#include <tuple>
#include <Eigen/Dense>

#include "gaussian.hpp"


namespace mtt {

template <typename Scalar, int DIM>
Eigen::Matrix<Scalar, DIM, DIM> symmetrize(const Eigen::Matrix<Scalar, DIM, DIM>& m) {
    return 0.5 * (m + m.transpose());
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
void kf_predict(Eigen::Matrix<Scalar, DIM, 1>& mu, Eigen::Matrix<Scalar, DIM, DIM>& cov,
                const Eigen::Matrix<Scalar, DIM, DIM>& F, const Eigen::Matrix<Scalar, DIM, DIM>& Q) {
    mu = F * mu;
    cov = symmetrize(F * cov * F.transpose() + Q);
}

template <typename Scalar, int DIM>
void kf_predict(Gaussian<Scalar, DIM>& gaus, const Eigen::Matrix<Scalar, DIM, DIM>& F,
                const Eigen::Matrix<Scalar, DIM, DIM>& Q) {
    kf_predict(gaus.mu, gaus.cov, F, Q);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
void kf_update(Eigen::Matrix<Scalar, SDIM, 1>& mu, Eigen::Matrix<Scalar, SDIM, SDIM>& cov, const Eigen::Matrix<Scalar, MDIM, 1>& z,
               const Eigen::Matrix<Scalar, MDIM, SDIM>& H, const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    using SST = Eigen::Matrix<Scalar, SDIM, SDIM>;
    using MMT = Eigen::Matrix<Scalar, MDIM, MDIM>;
    using SMT = Eigen::Matrix<Scalar, SDIM, MDIM>;

    MMT S = symmetrize(H * cov * H.transpose() + R);
    SMT A = cov * H.transpose();
    // K = A * S^-1 -> K^T = (S^-1)^T * A^T -> K^T = S^-1 * A^T -> S * K^T = A^T
    Eigen::LLT<MMT> llt(S);
    SMT K = llt.solve(A.transpose()).transpose();

    auto innov = z - H * mu;
    mu += K * innov;
    cov = symmetrize(cov - K * A.transpose());

    // MMT S = symmetrize(H * cov * H.transpose() + R);
    // SMT A = cov * H.transpose();
    // SMT K = A * S.inverse();
    // cov = symmetrize(cov - K * A.transpose());
    // auto innov = z - kf_update_eta(mu, H);
    // mu += K * innov;
}

template <typename Scalar, int SDIM, int MDIM>
void kf_update(Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, 1>& z,
               const Eigen::Matrix<Scalar, MDIM, SDIM>& H, const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    kf_update(gaus.mu, gaus.cov, z, H, R);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, 1> kf_eta(const Eigen::Matrix<Scalar, SDIM, 1>& mu, const Eigen::Matrix<Scalar, MDIM, SDIM>& H) {
    return H * mu;
}

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, 1> kf_eta(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, SDIM>& H) {
    return kf_eta(gaus.mu, H);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, MDIM> kf_S(const Eigen::Matrix<Scalar, SDIM, SDIM>& cov, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
                                           const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    return symmetrize(H * cov * H.transpose() + R);
}

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, MDIM> kf_S(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
                                           const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    return kf_S(gaus.cov, H, R);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, MDIM> kf_S_inv(const Eigen::Matrix<Scalar, SDIM, SDIM>& cov, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
                                           const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    using MMT = Eigen::Matrix<Scalar, MDIM, MDIM>;

    MMT S = symmetrize(H * cov * H.transpose() + R);
    Eigen::LLT<MMT> llt(S);
    return llt.solve(MMT::Identity());

    // return symmetrize((H * cov * H.transpose() + R).inverse());
}

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, MDIM, MDIM> kf_S_inv(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
                                           const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    return kf_S_inv(gaus.cov, H, R);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
std::pair<Eigen::Matrix<Scalar, MDIM, MDIM>, Scalar>
kf_S_inv_logdet(const Eigen::Matrix<Scalar, SDIM, SDIM>& cov, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
             const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    using MMT = Eigen::Matrix<Scalar, MDIM, MDIM>;

    MMT S = symmetrize(H * cov * H.transpose() + R);
    Eigen::LLT<MMT> llt(S);
    Scalar logdet = 0;
    auto& L = llt.matrixL();
    for (size_t i = 0; i < MDIM; i++)
        logdet += std::log(L(i, i));
    logdet *= 2;
    return {llt.solve(MMT::Identity()), logdet};
}

template <typename Scalar, int SDIM, int MDIM>
std::pair<Eigen::Matrix<Scalar, MDIM, MDIM>, Scalar>
kf_S_inv_logdet(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
             const Eigen::Matrix<Scalar, MDIM, MDIM>& R) {
    return kf_S_inv_logdet(gaus.cov, H, R);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
std::pair<Eigen::Matrix<Scalar, SDIM, SDIM>, Eigen::Matrix<Scalar, SDIM, MDIM>>
kf_cov_K(const Eigen::Matrix<Scalar, SDIM, SDIM>& cov, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
         const Eigen::Matrix<Scalar, MDIM, MDIM>& S_inv) {
    using SST = Eigen::Matrix<Scalar, SDIM, SDIM>;
    using SMT = Eigen::Matrix<Scalar, SDIM, MDIM>;

    SMT A = cov * H.transpose();
    SMT K = A * S_inv;
    SST P = symmetrize(cov - K * A.transpose());
    return {P, K};
}

template <typename Scalar, int SDIM, int MDIM>
std::pair<Eigen::Matrix<Scalar, SDIM, SDIM>, Eigen::Matrix<Scalar, SDIM, MDIM>>
kf_cov_K(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, SDIM>& H,
         const Eigen::Matrix<Scalar, MDIM, MDIM>& S_inv) {
    return kf_cov_K(gaus.cov, H, S_inv);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, SDIM, 1> kf_mean(const Eigen::Matrix<Scalar, SDIM, 1>& mu, const Eigen::Matrix<Scalar, MDIM, 1>& z,
                                       const Eigen::Matrix<Scalar, SDIM, MDIM>& K, const Eigen::Matrix<Scalar, MDIM, 1>& eta) {
    return mu + K * (z - eta);
}

template <typename Scalar, int SDIM, int MDIM>
Eigen::Matrix<Scalar, SDIM, 1> kf_mean(const Gaussian<Scalar, SDIM>& gaus, const Eigen::Matrix<Scalar, MDIM, 1>& z,
                                       const Eigen::Matrix<Scalar, SDIM, MDIM>& K, const Eigen::Matrix<Scalar, MDIM, 1>& eta) {
    return kf_mean(gaus.mu, z, K, eta);
}

} // namespace mtt
