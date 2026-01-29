#pragma once
#include <vector>
#include <Eigen/Dense>

#include "gaussian.hpp"


namespace mtt {

template <typename Scalar, int MDIM>
struct Gater {
    using MeasT = Eigen::Matrix<Scalar, MDIM, 1>;
    using MeanT = Eigen::Matrix<Scalar, MDIM, 1>;
    using CovT = Eigen::Matrix<Scalar, MDIM, MDIM>;

    std::vector<MeasT> measurements;
    std::vector<uint32_t> idxs;
    std::vector<Scalar> distances;

    constexpr Gater(const std::vector<MeasT>& measurements) : measurements(measurements) {
        idxs.reserve(size());
        distances.reserve(size());
    }
    constexpr Gater(std::vector<MeasT>&& measurements) : measurements(std::move(measurements)) {
        idxs.reserve(size());
        distances.reserve(size());
    }

    constexpr size_t gate_inv(const MeanT& mu, const CovT& cov_inv, Scalar max_std) {
        idxs.clear();
        for (size_t i = 0; i < measurements.size(); i++) {
            Scalar dist = mahalanobis_distance_sq(z, mu, cov_inv);
            if (mahalanobis_distance_sq(z, mu, cov_inv) <= max_std) {
                idxs.push_back(i);
                distances.push_back(dist);
            }
        }
        return idxs.size();
    }
    constexpr size_t gate(const MeanT& mu, const CovT& cov, Scalar max_std) {
        return gate_inv(mu, cov.inverse(), max_std);
    }
    constexpr size_t gate(const Gaussian<Scalar, MDIM>& gaus, Scalar max_std) {
        return gate_inv(gaus.mu, gaus.cov.inverse(), max_std);
    }
    
    constexpr Scalar mahalanobis_distance(const MeasT& x, const MeanT& mu, const CovT& cov_inv) const {
        return sqrt(mahalanobis_distance_sq(x, mu, cov_inv));
    }
    constexpr Scalar mahalanobis_distance_sq(const MeasT& x, const MeanT& mu, const CovT& cov_inv) const {
        MeanT dif = x - mu;
        return dif.transpose() * cov_inv * dif;
    }

    [[nodiscard]] constexpr size_t size() const noexcept {
        return measurements.size();
    }
    [[nodiscard]] constexpr size_t gate_size() const noexcept {
        return idxs.size();
    }
};

} // namespace mtt
