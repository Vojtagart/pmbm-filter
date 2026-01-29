#pragma once
#include <Eigen/Dense>


namespace mtt {

template <typename Scalar, int DIM>
struct Bernoulli {
    using MeanT = Eigen::Matrix<Scalar, DIM, 1>;
    using CovT = Eigen::Matrix<Scalar, DIM, DIM>;
    Scalar r;
    MeanT mu;
    CovT cov;

    Bernoulli() = default;
    template <typename M, typename C>
    constexpr Bernoulli(Scalar r, M&& mu, C&& cov) : r(r), mu(std::forward<M>(m)), cov(std::forward<C>(cov)) {}

    constexpr void swap(Gaussian& other) noexcept {
        using std::swap;
        swap(r, other.r);
        swap(mu, other.mu);
        swap(cov, other.cov);
    }
    friend constexpr void swap(Gaussian& lhs, Gaussian& rhs) noexcept {
        lhs.swap(rhs);
    }
};

} // namespace mtt
