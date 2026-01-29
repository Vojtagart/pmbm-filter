#pragma once
#include <Eigen/Dense>

#include "gaussian.hpp"


namespace mtt {

template <typename Scalar, int DIM>
struct GaussianMixture {
    using GaussT = Gaussian<Scalar, DIM>;
    std::vector<Scalar> W;
    std::vector<GaussT> G;

    constexpr void filter_out(Scalar min_w) {
        for (size_t i = 0; i < size();) {
            if (w(i) < min_w)
                erase(i);
            else
                i++;
        }
    }
    constexpr void erase(size_t idx) {
        assert(idx < size());
        using std::swap;
        swap(W[idx], W.back());
        swap(G[idx], G.back());
        W.pop_back();
        G.pop_back();
    }
    constexpr void scale_weight(Scalar val) {
        for (auto& x : W) {
            x *= val;
        }
    }
    constexpr void reserve(size_t cap) {
        W.reserve(cap);
        G.reserve(cap);
    }
    constexpr void push(Scalar w, const GaussT& g) {
        W.push_back(w);
        G.push_back(g);
    }
    constexpr void push(Scalar w, GaussT&& g) {
        W.push_back(w);
        G.push_back(std::move(g));
    }
    template <class... Args>
    constexpr void emplace(Scalar w, Args&&... args) {
        W.push_back(w);
        G.emplace_back(std::forward<Args>(args)...);
    }

    [[nodiscard]] constexpr size_t size() const noexcept {
        return W.size();
    }
    [[nodiscard]] constexpr Scalar& w(size_t idx) {
        assert(idx < size());
        return W[idx];
    }
    [[nodiscard]] constexpr Scalar w(size_t idx) const {
        assert(idx < size());
        return W[idx];
    }
    [[nodiscard]] constexpr GaussT::MeanT& mu(size_t idx) {
        assert(idx < size());
        return G[idx].mu;
    }
    [[nodiscard]] constexpr const GaussT::MeanT& mu(size_t idx) const {
        assert(idx < size());
        return G[idx].mu;
    }
    [[nodiscard]] constexpr GaussT::CovT& cov(size_t idx) {
        assert(idx < size());
        return G[idx].cov;
    }
    [[nodiscard]] constexpr const GaussT::CovT& cov(size_t idx) const {
        assert(idx < size());
        return G[idx].cov;
    }
    [[nodiscard]] constexpr GaussT& gauss(size_t idx) {
        assert(idx < size());
        return G[idx];
    }
    [[nodiscard]] constexpr const GaussT& gauss(size_t idx) const {
        assert(idx < size());
        return G[idx];
    }

    constexpr void swap(GaussianMixture& other) noexcept {
        using std::swap;
        swap(W, other.W);
        swap(G, other.G);
    }
    friend constexpr void swap(GaussianMixture& lhs, GaussianMixture& rhs) noexcept {
        lhs.swap(rhs);
    }
};

} // namespace mtt
