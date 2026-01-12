from pathlib import Path
import shutil
import sys

sys.path.append(".")

from nightly_build.stem_runners.pekeris import run_pekeris
from nightly_build.stem_runners.strip_load_2D import run_strip_2D
from nightly_build.stem_runners.strip_load_3D import run_strip_3D
from nightly_build.stem_runners.one_dim_wave_prop import run_column
# from nightly_build.stem_runners.moving_load_halfspace import run_moving_load
from nightly_build.stem_runners.sdof import run_sdof
from nightly_build.stem_runners.vibrating_dam import run_vibrating_dam
from nightly_build.stem_runners.vibrating_dam_3D import run_vibrating_dam_3d
from nightly_build.stem_runners.simply_supported_beam import run_simply_supported_beam

from nightly_build.stem_runners.compare_analytical import compare_pekeris, compare_strip_load_2D, \
      compare_strip_load_3D, compare_wave_propagation, compare_sdof, compare_vibrating_dam, compare_simply_supported_beam #, compare_moving_load


def move_file(src: Path, dest: Path):
    shutil.move(src, dest)


run_sdof(Path("./sdof_tmp"))
compare_sdof(r"sdof_tmp/sdof_results.json", r"nightly_build/sdof/time_history.pdf")
shutil.rmtree("sdof_tmp")

run_column(Path("./one_dim_wave_prop_tmp"))
compare_wave_propagation(r"one_dim_wave_prop_tmp/output/calculated_output.json",
                         r"nightly_build/one_dim_wave_prop/time_history.pdf")
shutil.rmtree("one_dim_wave_prop_tmp")

run_pekeris(Path("./lamb_tmp"))
compare_pekeris(r"lamb_tmp/json_output.json", r"nightly_build/lamb/time_history.pdf")
shutil.rmtree("lamb_tmp")

run_strip_2D(Path("./strip_2D_tmp"))
compare_strip_load_2D(r"strip_2D_tmp/output/output_vtk_porous_computational_model_part",
                      r"nightly_build/strip_load_2D/time_history.pdf")
shutil.rmtree("strip_2D_tmp")

run_strip_3D(Path("./strip_3D_tmp"))
compare_strip_load_3D(r"strip_3D_tmp/output/output_vtk_porous_computational_model_part",
                      r"nightly_build/strip_load_3D/time_history.pdf")
shutil.rmtree("strip_3D_tmp")

run_vibrating_dam(Path("./vibrating_dam_2D_tmp"))
compare_vibrating_dam(r"vibrating_dam_2D_tmp/json_output_top.json",
                      r"nightly_build/vibrating_dam/power_spectral_density.pdf")

shutil.rmtree("vibrating_dam_2D_tmp")

run_vibrating_dam_3d(Path("./vibrating_dam_3D_tmp"))
compare_vibrating_dam(r"vibrating_dam_3D_tmp/json_output_top.json",
                      r"nightly_build/vibrating_dam_3D/power_spectral_density.pdf")

shutil.rmtree("vibrating_dam_3D_tmp")

run_simply_supported_beam(Path("./simply_supported_beam_tmp"), 2)
run_simply_supported_beam(Path("./simply_supported_beam_tmp"), 3)

compare_simply_supported_beam(r"simply_supported_beam_tmp",
                              r"nightly_build/simply_supported_beam/simply_supported_beam_results.pdf")

shutil.rmtree("simply_supported_beam_tmp")

# run_moving_load(Path("./moving_load_halfspace_tmp"))
# compare_moving_load(r"moving_load_halfspace_tmp/output/calculated_output.json",
#                     r"nightly_build/moving_load_halfspace/time_history.pdf")
# shutil.rmtree("moving_load_halfspace_tmp")
