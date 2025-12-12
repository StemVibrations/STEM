from pathlib import Path
import shutil
import sys

sys.path.append(".")

from nightly_build.stem_runners.pekeris import run_pekeris
from nightly_build.stem_runners.strip_load_2D import run_strip_2D
from nightly_build.stem_runners.strip_load_3D import run_strip_3D
from nightly_build.stem_runners.one_dim_wave_prop import run_column

from nightly_build.stem_runners.compare_analytical import compare_pekeris, compare_strip_load_2D, \
      compare_strip_load_3D, compare_wave_propagation


def move_file(src: Path, dest: Path):
    shutil.move(src, dest)


run_pekeris(Path("./pekeris_tmp"))
run_strip_2D(Path("./strip_2D_tmp"))
run_strip_3D(Path("./strip_3D_tmp"))
run_column(Path("./one_dim_wave_prop_tmp"))

# compare_analytical
compare_pekeris(r"pekeris_tmp/json_output.json", r"nightly_build/pekeris/time_history.pdf")
compare_strip_load_2D(r"strip_2D_tmp/output/output_vtk_porous_computational_model_part",
                      r"nightly_build/strip_load_2D/time_history.pdf")
compare_strip_load_3D(r"strip_3D_tmp/output/output_vtk_porous_computational_model_part",
                      r"nightly_build/strip_load_3D/time_history.pdf")
compare_wave_propagation(r"one_dim_wave_prop_tmp/output/calculated_output.json",
                         r"nightly_build/one_dim_wave_prop/time_history.pdf")

# delete all tmp folders
# shutil.rmtree("pekeris_tmp")
# shutil.rmtree("strip_2D_tmp")
# shutil.rmtree("strip_3D_tmp")
# shutil.rmtree("one_dim_wave_prop_tmp")