import argparse
import subprocess
import sys
import sysconfig
from pathlib import Path

import mlx
import nanobind

r = Path(__file__).parent
m = next((Path(p) for p in mlx.__path__ if (Path(p) / "include").exists()), Path(mlx.__path__[0]))
n = Path(nanobind.__file__).parent
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=r.parent / "src/rec_mlx")
a = parser.parse_args()
out = a.output / ("_grammar_native" + sysconfig.get_config_var("EXT_SUFFIX"))
is_darwin = sys.platform == "darwin"
cmd = [
    "clang++" if is_darwin else "c++",
    "-std=c++20",
    "-DNB_DOMAIN=mlx",
    "-O3",
    "-shared",
    "-fPIC",
    "-fvisibility=hidden",
]
if is_darwin:
    cmd += ["-undefined", "dynamic_lookup", "-Wl,-rpath,@loader_path/../mlx/lib"]
else:
    cmd += ["-Wl,-rpath,$ORIGIN/../mlx/lib"]
cmd += [
    "-I" + str(m / "include"),
    "-I" + str(n / "include"),
    "-I" + str(n / "ext/robin_map/include"),
    "-I" + sysconfig.get_path("include"),
    str(r / "grammar_native.cpp"),
    str(n / "src/nb_combined.cpp"),
    "-L" + str(m / "lib"),
    "-lmlx",
    "-o",
    str(out),
]
subprocess.run(cmd, check=True)
print(out)
