from helpers import campanella, furnsh_kernels


def test_import():
    furnsh_kernels()
    for name in ["SpiceParameters", "IvpResult", "solve_ivp_rv", "solve_ivp_mee", "solve_ivpstm_rv",
                 "solve_ivpstm_mee", "eom_rv", "eom_mee", "eomstm_rv", "eomstm_mee", "eval_rv",
                 "eval_mee", "furnsh", "mee2cart", "cart2mee", "mee2cart_jacobian",
                 "load_spherical_harmonics", "sh_accel_bf", "sh_accel_inertial",
                 "ThrustFrame", "ThrustOptions", "solve_ivp_rvm", "solve_ivp_meem", "solve_ivpstm_rvm",
                 "solve_ivpstm_meem", "eval_rvm", "eval_meem", "eom_rvm", "eom_meem", "eomstm_rvm",
                 "eomstm_meem", "propagate_zoh_rvm", "propagate_zoh_meem", "ZohLinearization"]:
        assert hasattr(campanella, name), name
    for name in ["PropagatorTwoBody", "PropagatorNbody", "PropagatorNbodySrp", "PropagatorGve",
                 "add_j2000", "ControlledParameters"]:
        assert not hasattr(campanella, name), name
