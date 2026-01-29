#pragma once
#include <cmath>
#include <numbers>
#include <Eigen/Dense>


namespace mtt {

template <typename Scalar, int DIM>
struct Gaussian {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    MeanT mu;
    CovT cov;

    Gaussian() = default;
    template <typename M, typename C>
    constexpr Gaussian(M&& mu, C&& cov) : mu(std::forward<M>(m)), cov(std::forward<C>(cov)) {}

    constexpr void swap(Gaussian& other) noexcept {
        using std::swap;
        swap(mu, other.mu);
        swap(cov, other.cov);
    }
    friend constexpr void swap(Gaussian& lhs, Gaussian& rhs) noexcept {
        lhs.swap(rhs);
    }

    // for some reason, std::log isn't constexpr...
    // constant term -0.5 * n * log(2pi)
    static const Scalar HALF_N_LOG_2PI = Scalar(0.5) * static_cast<Scalar>(DIM) * std::log(2 * std::numbers::pi_v<Scalar>);
    static const Scalar TWOPI_TO_N_HALF = std::pow(2 * std::numbers::pi_v<Scalar>, static_cast<Scalar>(DIM) * Scalar(0.5));
};

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
Scalar mahalanobis_distance_inv(const Eigen::Matrix<Scalar, DIM, 1>& x, const Eigen::Matrix<Scalar, DIM, 1>& mu,
                                const Eigen::Matrix<Scalar, DIM, DIM>& cov_inv) {
    auto dif = (x - mu).eval();
    return dif.transpose() * cov_inv * dif;
}

template <typename Scalar, int DIM>
Scalar mahalanobis_distance(const Eigen::Matrix<Scalar, DIM, 1>& x, const Eigen::Matrix<Scalar, DIM, 1>& mu,
                            const Eigen::Matrix<Scalar, DIM, DIM>& cov) {
    return mahalanobis_distance_inv(x, mu, cov.inverse())
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
Scalar mvn_pdf(const Eigen::Matrix<Scalar, DIM, 1>& x, const Eigen::Matrix<Scalar, DIM, 1>& mu,
               const Eigen::Matrix<Scalar, DIM, DIM>& cov) {
    using GaussianT = Gaussian<Scalar, DIM>;

    Scalar norm_factor = Scalar(1) / (GaussinaT::TWOPI_TO_N_HALF * std::sqrt(cov.determinant()));
    Scalar qnorm = Scalar(-0.5) * mahalanobis_distance(x, mu, cov);
    return norm_factor * std::exp(qnorm);
}

template <typename Scalar, int DIM>
Scalar mvn_pdf(const Eigen::Matrix<Scalar, DIM, 1>& x, const Gaussian<Scalar, DIM>& gaus) {
    return mvn_pdf(x, gaus.mu, gaus.cov);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
Scalar mvn_logpdf(const Eigen::Matrix<Scalar, DIM, 1>& x, const Eigen::Matrix<Scalar, DIM, 1>& mu,
                  const Eigen::Matrix<Scalar, DIM, DIM>& cov) {
    using GaussianT = Gaussian<Scalar, DIM>;

    Scalar norm_factor = -GaussianT::HALF_N_LOG_2PI - Scalar(0.5) * std::log(cov.determinant());
    Scalar qnorm = Scalar(-0.5) * mahalanobis_distance(x, mu, cov);
    return norm_factor + qnorm;
}

template <typename Scalar, int DIM>
Scalar mvn_logpdf(const Eigen::Matrix<Scalar, DIM, 1>& x, const Gaussian<Scalar, DIM>& gaus) {
    return mvn_logpdf(x, gaus.mu, gaus.cov);
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
Scalar mvn_logpdf_from_c_mahal(Scalar c, Scalar mahal_dist) {
    return c - Scalar(0.5) * mahal_dist;
}

template <typename Scalar, int DIM>
Scalar mvn_logc_from_logdet(Scalar logdet) {
    return -GaussianT::HALF_N_LOG_2PI - Scalar(0.5) * logdet;
}

//------------------------------------------------------------------------------------

template <typename Scalar, int DIM>
struct MvnPdf {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    using GaussianT = Gaussian<Scalar, DIM>;

    Scalar c;
    MeanT mu;
    MeanT cov_inv;

    MvnPdf(const MeanT& mu, const MeanT& cov)
            : c(Scalar(1) / (GaussianT::TWOPI_TO_N_HALF * std::sqrt(cov.determinant()))),
              mu(mu), cov_inv(cov.inverse()) {}
    MvnPdf(const GaussianT& gaus) : MvnPdf(gaus.mu, gaus.cov) {}
    constexpr Scalar operator() (const MeanT& x) const {
        Scalar qnorm = Scalar(-0.5) * mahalanobis_distance_inv(x, mu, cov_inv);
        return c * std::exp(qnorm);
    }
};

template <typename Scalar, int DIM>
struct MvnLogPdf {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    using GaussianT = Gaussian<Scalar, DIM>;

    Scalar c;
    MeanT mu;
    MeanT cov_inv;

    MvnLogPdf(const MeanT& mu, const MeanT& cov)
            : c(-GaussianT::HALF_N_LOG_2PI - Scalar(0.5) * std::log(cov.determinant())),
              mu(mu), cov_inv(cov.inverse()) {}
    MvnLogPdf(const GaussianT& gaus) : MvnLogPdf(gaus.mu, gaus.cov) {}
    constexpr Scalar operator() (const MeanT& x) const {
        Scalar qnorm = Scalar(-0.5) * mahalanobis_distance_inv(x, mu, cov_inv);
        return c + qnorm;
    }
};

//------------------------------------------------------------------------------------

} // namespace mtt
