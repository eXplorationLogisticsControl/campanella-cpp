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


np.set_printoptions(precision=3)

import campanella


def test_propagate_NbodySRP():
    spice_dir = os.environ["SPICE"]
    campanella.furnsh(os.path.join(spice_dir, "lsk", "naif0012.tls"))
    campanella.furnsh(os.path.join(spice_dir, "spk", "de440.bsp"))
    
    et0 = 740463475.5117936
    state0 = [-7505.7613773288285, 44128.19419987408, -55792.327508873765,
              0.07193326412942684, 0.018407794390508374, 0.004868437733535174]
    tspan = [0.0, 20*86400.0]
    GM = 4.902800582147760e+03
    GM_list = [GM, 398600.432896939, 132712197035.766]
    GM = 398600.44
    print("Initializing propagator for two-body problem")
    prop = campanella.PropagatorTwoBody(GM)
    traj = prop.solve_ivp(
        tspan = tspan,
        y0 = state0,
    )
    print(f"Final state:\n {traj[-1]}")
    print(f"np.array(traj).shape: \n{np.array(traj).shape}")

    print("Initializing N-body propagator...")
    propN = campanella.PropagatorNBodySRP(
        GM = GM_list,
        BODIES = ["MOON", "EARTH", "SUN"],
        frame = "J2000",
        A = 34.675,
        m = 1850,
        rhoS = 0.9,
        rhoD = 0,
        LU = 1.0,
        TU = 1.0,
    )
    print("Propagating...")
    trajN = propN.solve_ivp(
        et0,
        tspan,
        state0
        # et0 = 946728069.3271508,
        # tspan = tspan,
        # y0 = state0,
    )
    print(f"Final state:\n {trajN[-1]}")

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    ax.plot(np.array(traj)[:,1], np.array(traj)[:,2], np.array(traj)[:,3])
    ax.plot(np.array(trajN)[:,1], np.array(trajN)[:,2], np.array(trajN)[:,3])

if __name__=="__main__":
    test_propagate_NbodySRP()
    print("Done!")
    plt.show()