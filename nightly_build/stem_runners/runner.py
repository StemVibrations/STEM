from pathlib import Path
import shutil
import sys

sys.path.append(".")

from nightly_build.stem_runners.pekeris import run_pekeris
from nightly_build.stem_runners.strip_load_2D import run_strip_2D
from nightly_build.stem_runners.strip_load_3D import run_strip_3D
from nightly_build.stem_runners.one_dim_wave_prop import run_column
from nightly_build.stem_runners.moving_load_halfspace import run_moving_load
from nightly_build.stem_runners.sdof import run_sdof
from nightly_build.stem_runners.boussinesq import run_boussinesq
from nightly_build.stem_runners.vibrating_dam import run_vibrating_dam
from nightly_build.stem_runners.vibrating_dam_3D import run_vibrating_dam_3d
from nightly_build.stem_runners.one_dim_abs_boundary import run_abs_boundary
from nightly_build.stem_runners.simply_supported_beam import run_simply_supported_beam
from nightly_build.stem_runners.moving_load_on_beam import run_moving_load_on_beam

from nightly_build.stem_runners.compare_analytical import compare_pekeris, compare_strip_load, \
      compare_wave_propagation, compare_sdof, compare_vibrating_dam, compare_abs_boundary, \
      compare_simply_supported_beam, compare_boussinesq, compare_moving_load_on_beam, compare_moving_load


def move_file(src: Path, dest: Path):
    """
    Move a file from src to dest.

    Args:
        - src (Path): Path to the source file.
        - dest (Path): Path to the destination file.
    """
    shutil.move(src, dest)


run_sdof(Path("./sdof_tmp"))
compare_sdof(Path("sdof_tmp/sdof_results.json"), Path("nightly_build/sdof/time_history.pdf"))
shutil.rmtree(Path("sdof_tmp"))

run_column(Path("./one_dim_wave_prop_tmp"), 2)
run_column(Path("./one_dim_wave_prop_tmp"), 3)
compare_wave_propagation(Path("one_dim_wave_prop_tmp/output"), Path("nightly_build/one_dim_wave_prop/time_history.pdf"))
shutil.rmtree(Path("one_dim_wave_prop_tmp"))

run_pekeris(Path("./lamb_tmp"))
compare_pekeris(Path("lamb_tmp/json_output.json"), Path("nightly_build/lamb/time_history.pdf"))
shutil.rmtree(Path("lamb_tmp"))

run_strip_2D(Path("./strip_2D_tmp"))
run_strip_3D(Path("./strip_3D_tmp"))
compare_strip_load([Path("strip_2D_tmp"), Path("strip_3D_tmp")], Path("nightly_build/strip_load/time_history.pdf"))
shutil.rmtree(Path("strip_2D_tmp"))
shutil.rmtree(Path("strip_3D_tmp"))

run_boussinesq(Path("./boussinesq_tmp"))
compare_boussinesq(Path("boussinesq_tmp"), Path("nightly_build/boussinesq/boussinesq_comparison.pdf"))
shutil.rmtree(Path("boussinesq_tmp"))

run_vibrating_dam(Path("./vibrating_dam_2D_tmp"))
run_vibrating_dam_3d(Path("./vibrating_dam_3D_tmp"))
compare_vibrating_dam([
    Path("vibrating_dam_2D_tmp/json_output_top.json"),
    Path("vibrating_dam_3D_tmp/json_output_top.json"),
], Path("nightly_build/vibrating_dam/power_spectral_density.pdf"))
shutil.rmtree(Path("vibrating_dam_2D_tmp"))
shutil.rmtree(Path("vibrating_dam_3D_tmp"))

run_abs_boundary(Path("./one_dim_abs_boundary_tmp"), 2)
run_abs_boundary(Path("./one_dim_abs_boundary_tmp"), 3)
compare_abs_boundary(Path("one_dim_abs_boundary_tmp/output"), Path("nightly_build/one_d_abs_boundary/time_history.pdf"))
shutil.rmtree(Path("one_dim_abs_boundary_tmp"))

run_simply_supported_beam(Path("./simply_supported_beam_tmp"), 2)
run_simply_supported_beam(Path("./simply_supported_beam_tmp"), 3)
compare_simply_supported_beam(Path("simply_supported_beam_tmp"),
                              Path("nightly_build/simply_supported_beam/simply_supported_beam_results.pdf"))
shutil.rmtree(Path("simply_supported_beam_tmp"))

run_moving_load_on_beam(Path("./moving_load_on_beam_tmp"), 2)
run_moving_load_on_beam(Path("./moving_load_on_beam_tmp"), 3)
compare_moving_load_on_beam(Path("moving_load_on_beam_tmp"),
                            Path("nightly_build/moving_load_on_beam/moving_load_on_beam_results.pdf"))
shutil.rmtree(Path("moving_load_on_beam_tmp"))

run_moving_load(Path("./moving_load_halfspace_tmp"))
compare_moving_load(Path("moving_load_halfspace_tmp/output/calculated_output_stage_2.json"),
                    Path("nightly_build/moving_load_halfspace/time_history.pdf"))
shutil.rmtree(Path("moving_load_halfspace_tmp"))
