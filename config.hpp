#pragma once
#include <Eigen/Dense>

#include "mtt_utils/mtt_utils.hpp"


namespace pmbm {

    //==========================================
    // Main config
    //==========================================

    /// @brief Dimension of the State vector
    inline constexpr int STATE_DIM = 4;

    /// @brief @brief Dimension of the measurement vector
    inline constexpr int MEAS_DIM = 2;

    /// @brief Scalar used for computations
    using Scalar = double;

    //==========================================
    // Aliases for Eigen types
    //==========================================

    /// @brief Type for vector representing state
    /// For example state or state estimates
    using StateT = Eigen::Matrix<Scalar, STATE_DIM, 1>;

    /// @brief Type for vector representing measurement
    /// For example measurement or innovation
    using MeasT = Eigen::Matrix<Scalar, MEAS_DIM, 1>;

    /// @brief Type used for state covariance matrix
    /// for example Process noise or variances inside Bernoullis
    using StateCovT = Eigen::Matrix<Scalar, STATE_DIM, STATE_DIM>;

    /// @brief Type used for measurement covariance matrix
    /// for example Measurement noise or Innovation covariance
    using MeasCovT = Eigen::Matrix<Scalar, MEAS_DIM, MEAS_DIM>;

    /// @brief Type used for Transition matrix
    using TransT = Eigen::Matrix<Scalar, STATE_DIM, STATE_DIM>;

    /// @brief Type used for Measurement matrix
    using MeasMatT = Eigen::Matrix<Scalar, MEAS_DIM, STATE_DIM>;

    /// @brief Type used for Kalman gain
    using KalmanGT = Eigen::Matrix<Scalar, STATE_DIM, MEAS_DIM>;

    //==========================================
    // Aliases for mtt_utils types
    //==========================================

    using StateGaussT = mtt::Gaussian<Scalar, STATE_DIM>;
    using MeasGaussT = mtt::Gaussian<Scalar, MEAS_DIM>;

    using StateGaussMixT = mtt::GaussianMixture<Scalar, STATE_DIM>;
    using MeasGaussMixT = mtt::GaussianMixture<Scalar, MEAS_DIM>;

    using StateBernT = mtt::Bernoulli<Scalar, STATE_DIM>;
    using StateBernT = mtt::Bernoulli<Scalar, MEAS_DIM>;

    using StateBernMixT = mtt::BernoulliMixture<Scalar, STATE_DIM>;
    using MeasBernMixT = mtt::BernoulliMixture<Scalar, MEAS_DIM>;
    

} // namespace pmbm