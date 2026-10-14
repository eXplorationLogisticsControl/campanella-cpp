import sys
import os

if sys.platform == "win32":
    load_dir = "build/Release"
else:
    load_dir = "build/lib"

sys.path.append(os.path.join(os.path.dirname(os.path.realpath(__file__)), "..", load_dir))


import campanella

def furnsh_kernels():
    spice_dir = os.environ["SPICE"]
    campanella.furnsh(os.path.join(spice_dir, "lsk", "naif0012.tls"))
    campanella.furnsh(os.path.join(spice_dir, "spk", "de440.bsp"))

def test_import():
    furnsh_kernels()
    campanella.add_j2000(2, 3)
    assert True


if __name__=="__main__":
    test_import()
    print("Done!")