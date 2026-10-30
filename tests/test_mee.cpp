// MEE conversions (no kernels needed)

#include <cmath>
#include <stdexcept>
#include <vector>

#include "helpers.hpp"

using namespace campanella_test;

namespace {

constexpr double MU_EARTH = 398600.4418;
const double PI = std::acos(-1.0);

}  // namespace

TEST(Mee, CartMeeCartRoundTrip) {
    const std::vector<std::vector<double>> states = {
        {-2103.7154800484846, 3643.742096113127, 5308.456731666615,
         -6.647330004445163, -3.833624329580397, 0.004603821856452162},      // LEO
        {5878.595087575521, 3000.5849397194843, -439.2672278754225,
         -3.6936046189819858, 8.180536450190633, -4.859712456831758},        // GTO
        {-55.46704301173019, -7246.552606392538, 896.5989572886389,
         -7.530175620232078, -1.4387377569063755, -0.7594926786100786},      // i = 170 deg
        {5775.472866359693, 7498.003270192943, 5354.4553621173945,
         -6.119856157164796, 5.295429592561897, 5.2676006404103894},         // hyperbolic
    };
    for (const auto& rv : states) {
        const std::vector<double> mee = campanella::cart2mee(rv, MU_EARTH);
        EXPECT_GT(mee[5], -PI);
        EXPECT_LE(mee[5], PI);
        const std::vector<double> back = campanella::mee2cart(mee, MU_EARTH);
        for (int i = 0; i < 6; i++) {
            EXPECT_NEAR(back[i], rv[i], 1e-12 * std::abs(rv[i < 3 ? 0 : 3]) + 1e-9) << i;
        }
    }
}

TEST(Mee, MoonReference) {
    const std::vector<double> expect = {5778.190022842265, -0.34651609849537257, 0.8514048990211349,
                                        1.9784209071365653, 0.6178109804659669, -1.184288478833202};
    const std::vector<double> mee = campanella::cart2mee(STATE0, MU_MOON);
    EXPECT_NEAR(mee[0], expect[0], 1e-13 * expect[0]);
    for (int i = 1; i < 6; i++) {
        EXPECT_NEAR(mee[i], expect[i], 1e-13);
    }
}

TEST(Mee, Singularities) {
    EXPECT_THROW(campanella::cart2mee({7000.0, 0.0, 0.0, 0.0, -7.5, 0.0}, MU_EARTH), std::domain_error);
    EXPECT_THROW(campanella::cart2mee({7000.0, 0.0, 0.0, 7.5, 0.0, 0.0}, MU_EARTH), std::domain_error);
    EXPECT_THROW(campanella::mee2cart({0.0, 0.1, 0.0, 0.0, 0.0, 0.0}, MU_EARTH), std::domain_error);
    EXPECT_THROW(campanella::mee2cart({7000.0, 1.5, 0.0, 0.0, 0.0, PI}, MU_EARTH), std::domain_error);
}

TEST(Mee, JacobianVsFiniteDifferences) {
    const std::vector<double> x = campanella::cart2mee(STATE0, MU_MOON);
    const std::vector<double> J = campanella::mee2cart_jacobian(x, MU_MOON);
    const std::vector<double> J_fd = jacobian_fd(
        [](const std::vector<double>& m) { return campanella::mee2cart(m, MU_MOON); }, x,
        {1e-6 * x[0], 1e-7, 1e-7, 1e-7, 1e-7, 1e-7});
    // column-normalised: columns have different units
    for (int j = 0; j < 6; j++) {
        double diff = 0.0, ref = 0.0;
        for (int i = 0; i < 6; i++) {
            diff = std::max(diff, std::abs(J[6 * i + j] - J_fd[6 * i + j]));
            ref = std::max(ref, std::abs(J_fd[6 * i + j]));
        }
        EXPECT_LE(diff / ref, 1e-6) << "column " << j;
    }
}
