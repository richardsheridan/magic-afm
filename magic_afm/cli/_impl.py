import enum
import pathlib

import click

from ..data_readers import SUFFIX_FVFILE_MAP
from ..calculation import FitMode


class TraceChoice(enum.IntEnum):
    TRACE = 0
    RETRACE = 1
    BOTH = 2
    ALL = -1


readable_file = click.Path(
    exists=False, dir_okay=False, readable=True, path_type=pathlib.Path
)


def abs_cb(c, p, v):
    return abs(v)


def readjson(c, p, options_json):
    if options_json is not None:
        import json

        with options_json:
            options_json = json.load(options_json)
    return options_json


def clip(lo, hi):
    def clip_inner(c, p, v):
        return min(max(v, lo), hi)

    return clip_inner


def suffix(c, p, filenames):
    unknown = []
    for filename in filenames:
        lower_suffix = filename.suffix.lower()
        if lower_suffix not in SUFFIX_FVFILE_MAP:
            if not lower_suffix[1:].isdigit():  # maybe nanoscope
                unknown.append(filename.name)
    if unknown:
        raise ValueError(f"Unknown filetypes for {unknown}")
    return filenames


@click.command()
@click.option(
    "--fit-mode",
    type=click.Choice(FitMode, case_sensitive=False),
    default=FitMode.RETRACT,
)
@click.option(
    "--trace",
    type=click.Choice(TraceChoice, case_sensitive=False),
    default=TraceChoice.TRACE,
)
@click.option("--k", type=float)
@click.option("--defl-sens", type=float)
@click.option("--sync-dist", type=float)
@click.option("+radius/-radius", "radius_flag", default=False)
@click.option("--radius", type=float, default=20.0, callback=abs_cb)
@click.option("+tau/-tau", "tau_flag", default=False)
@click.option("--tau", type=float, default=0.0, callback=clip(0.0, 1.0))
@click.option("+lj-scale/-lj-scale", "lj_scale_flag", default=True)
@click.option("--lj-scale", type=float, default=2.0, callback=clip(-6.0, 6.0))
@click.option("+vd/-vd", "vd_flag", default=False)
@click.option("--vd", type=float, default=0.0)
@click.option("+li_per/-li_per", "li_per_flag", default=False)
@click.option("--li_per", type=float, default=0.0, callback=abs_cb)
@click.option("+li_amp/-li_amp", "li_amp_flag", default=False)
@click.option("--li_amp", type=float, default=0.0, callback=abs_cb)
@click.option("+drag/-drag", "drag_flag", default=False)
@click.option("--drag", type=float, default=0.0, callback=abs_cb)
@click.option("--options-json", type=click.File("rb"), callback=readjson)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.argument(
    "filenames", nargs=-1, type=readable_file, required=True, callback=suffix
)
def main(
    fit_mode,
    output_path,
    k,
    defl_sens,
    sync_dist,
    trace,
    radius,
    radius_flag,
    tau,
    tau_flag,
    lj_scale,
    lj_scale_flag,
    vd,
    vd_flag,
    li_per,
    li_per_flag,
    li_amp,
    li_amp_flag,
    drag,
    drag_flag,
    options_json,
    filenames,
):
    """test docstring

    more docstring"""

    # prepare output folders early
    if output_path is None:
        pass
    elif output_path.is_absolute():
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        for filename in filenames:
            (filename / output_path).mkdir(parents=True, exist_ok=True)

    click.echo(
        (
            fit_mode,
            output_path,
            k,
            defl_sens,
            sync_dist,
            trace,
            radius,
            radius_flag,
            tau,
            tau_flag,
            lj_scale,
            lj_scale_flag,
            vd,
            vd_flag,
            li_per,
            li_per_flag,
            li_amp,
            li_amp_flag,
            drag,
            drag_flag,
            options_json,
            filenames,
        )
    )
