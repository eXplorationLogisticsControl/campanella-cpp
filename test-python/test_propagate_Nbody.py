"""
Test for propagating Nbody problem
"""

import copy
import os
import numpy as np
import sys
import matplotlib.pyplot as plt

if sys.platform == "win32":
    load_dir = "build/Release"
else:
    load_dir = "build/lib"

sys.path.append(
    os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "..", load_dir
    )
)

np.set_printoptions(precision=4)

import campanella


def furnsh_kernels():
    spice_dir = os.environ["SPICE"]
    campanella.furnsh(os.path.join(spice_dir, "lsk", "naif0012.tls"))
    campanella.furnsh(os.path.join(spice_dir, "spk", "de440.bsp"))


def test_twobody():
    """Test for propagating state in two-body dynamics"""
    et0 = 740463475.5117936
    state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
              0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
    tspan = [0.0, 20*86400.0]
    GM = 4.902800582147760e+03
    GM_list = [GM, 398600.432896939, 132712197035.766]

    #print("Initializing propagator for two-body problem")
    prop = campanella.PropagatorTwoBody(GM)
    traj = prop.solve_ivp(
        tspan = tspan,
        y0 = state0,
    )
    # print(f"Final state:\n {traj[-1]}")
    # print(f"np.array(traj).shape: \n{np.array(traj).shape}")
    tx_final_expect = np.array([1728000.0, -18205.698468519975,
                                26479.25761849232,
                                -38615.225993310676, 0.01263483050054705,
                                0.1635690404557372, -0.19163703441809382])
    assert np.allclose(np.array(traj[-1]), tx_final_expect, rtol=1e-12)


def test_propagate_Nbody(make_plots = False):
    """Test for propagating state & STM"""
    furnsh_kernels()
    et0 = 740463475.5117936
    state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
              0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
    tspan = [0.0, 20*86400.0]
    GM = 4.902800582147760e+03
    GM_list = [GM, 398600.432896939, 132712197035.766]

    # print("Initializing N-body propagator...")
    propN = campanella.PropagatorNbody(
        GM = GM_list,
        BODIES = ["MOON", "EARTH", "SUN"],
        frame = "J2000",
        LU = 1.0,
        TU = 1.0,
    )
    trajN = propN.solve_ivp(
        et0,
        tspan,
        state0
    )
    print(f"Final state:\n   {trajN[-1]}")
    tx_final_expect = np.array([1728000.0,
                                10171.343017143072,
                                33674.15294325228,
                                -60877.5267967639,
                                0.013967106146233203,
                                -0.0805131196626756,
                                0.0033270015062299776])
    print(f"np.array(trajN[-1]) = {np.array(trajN[-1])}")
    assert np.allclose(np.array(trajN[-1]), tx_final_expect, rtol=1e-12),\
        f"trajN[-1] = {trajN[-1]}, {trajN[0]}"
    
    # also propagate STM
    print("Also propagate STM...")
    trajN = propN.solve_ivp_stm(
        et0,
        tspan,
        np.concatenate((state0, np.eye(6).flatten()))
    )
    tf = trajN[-1][0]
    statef = trajN[-1][1:]
    statef = np.concatenate((np.array(statef[0:3]), np.array(statef[3:6])))
    print(f"Final state:\n   {tf}, {list(statef)} [km,km/s]")
    stmf = np.array(trajN[-1][7:]).reshape(6,6)
    print(f"STM final:\n {stmf}")

    # check against numerical STM via central differencing
    STM_num = np.zeros((6,6))
    h_ptrb = 1e-6
    for idx in range(6):
        state0_ptrb = copy.deepcopy(state0)
        state0_ptrb[idx] += h_ptrb
        traj_ptrb_forward = propN.solve_ivp(
            et0,
            tspan,
            state0_ptrb
        )

        state0_ptrb = copy.deepcopy(state0)
        state0_ptrb[idx] -= h_ptrb
        traj_ptrb_backward = propN.solve_ivp(
            et0,
            tspan,
            state0_ptrb
        )
        STM_num[:,idx] = (np.array(traj_ptrb_forward[-1][1:7]) - np.array(traj_ptrb_backward[-1][1:7])) / (2*h_ptrb)
    print(f"\nSTM_num:\n{STM_num}\n")

    diff = STM_num - stmf
    print(f"diff:\n{diff}")
    assert np.allclose(STM_num, stmf, rtol=1e-2),\
        f"STM_num:\n{STM_num}\n\nstmf:\n{stmf}\n\ndiff:\n{diff}"

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(np.array(trajN)[:,1], np.array(trajN)[:,2], np.array(trajN)[:,3])
    assert True


def test_propagate_Nbody_canonical(make_plots = False):
    """Test for propagating state & STM in canonical scales"""
    furnsh_kernels()
    et0 = 740463475.5117936
    state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
              0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
    tspan = [0.0, 20*86400.0]
    GM = 4.902800582147760e+03
    GM_list = [GM, 398600.432896939, 132712197035.766, 1.2671276409999998E+08]
    BODIES = ["301", "399", "10", "5"]

    # define canonical scales
    LU = 3000.0
    VU = np.sqrt(GM/LU)
    TU = LU/VU
    state0_canonical = np.concatenate((np.array(state0[:3])/LU, np.array(state0[3:])/VU))
    tspan_canonical  = list(np.array(tspan)/TU)
    GM_canonical = list(np.array(GM_list)/GM)

    print("Initializing N-body propagator (canonical)...")
    propN = campanella.PropagatorNbody(
        GM = GM_canonical,
        BODIES = BODIES,
        frame = "J2000",
        LU = LU,
        TU = TU,
    )
    trajN = propN.solve_ivp(
        et0,
        tspan_canonical,
        state0_canonical
    )
    tf = trajN[-1][0] * TU
    statef = trajN[-1][1:]
    statef = np.concatenate((np.array(statef[0:3])*LU, np.array(statef[3:6])*VU))
    print(f"Final state:\n   {tf}, {list(statef)}")

    # also propagate STM
    print("Also propagate STM...")
    trajN = propN.solve_ivp_stm(
        et0,
        tspan_canonical,
        np.concatenate((state0_canonical, np.eye(6).flatten()))
    )
    tf = trajN[-1][0] * TU
    statef = trajN[-1][1:]
    statef = np.concatenate((np.array(statef[0:3])*LU, np.array(statef[3:6])*VU))
    print(f"Final state:\n   {tf}, {list(statef)} [LU,VU]")
    stmf = np.array(trajN[-1][7:]).reshape(6,6)
    print(f"STM final:\n {stmf}")

    # check against numerical STM via central differencing
    STM_num = np.zeros((6,6))
    h_ptrb = 1e-6
    for idx in range(6):
        state0_ptrb = copy.deepcopy(state0_canonical)
        state0_ptrb[idx] += h_ptrb
        traj_ptrb_forward = propN.solve_ivp(
            et0,
            tspan_canonical,
            state0_ptrb
        )

        state0_ptrb = copy.deepcopy(state0_canonical)
        state0_ptrb[idx] -= h_ptrb
        traj_ptrb_backward = propN.solve_ivp(
            et0,
            tspan_canonical,
            state0_ptrb
        )
        STM_num[:,idx] = (np.array(traj_ptrb_forward[-1][1:7]) - np.array(traj_ptrb_backward[-1][1:7])) / (2*h_ptrb)
    print(f"\nSTM_num:\n{STM_num}\n")

    diff = STM_num - stmf
    print(f"diff:\n{diff}")
    assert np.allclose(STM_num, stmf, rtol=1e-4),\
        f"STM_num:\n{STM_num}\n\nstmf:\n{stmf}\n\ndiff:\n{diff}"

    if make_plots:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot(np.array(trajN)[:,1], np.array(trajN)[:,2], np.array(trajN)[:,3])
    assert True 


if __name__=="__main__":
    test_twobody()
    test_propagate_Nbody(make_plots = True)
    test_propagate_Nbody_canonical(make_plots = True)

    print("Done!")
    plt.show()