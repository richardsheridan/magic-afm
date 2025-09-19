import enum
import pathlib

import click


from ..data_readers import SUFFIX_FVFILE_MAP
from ..calculation import FitMode, FitFix


class TraceChoice(enum.IntEnum):
    TRACE = 0
    RETRACE = 1
    BOTH = 2
    ALL = -1


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
@click.option("-fix-radius/-fit-radius", default=None)
@click.option("--radius", type=float, default=20.0, callback=abs_cb)
@click.option("-fix-tau/-fit-tau", default=None)
@click.option("--tau", type=float, default=0.0, callback=clip(0.0, 1.0))
@click.option("-fix-lj-scale/-fit-lj-scale", default=None)
@click.option("--lj-scale", type=float, default=2.0, callback=clip(-6.0, 6.0))
@click.option("-fix-vd/-fit-vd", default=None)
@click.option("--vd", type=float, default=0.0)
@click.option("-fix-li-per/-fit-li-per", default=None)
@click.option("--li-per", type=float, default=0.0, callback=abs_cb)
@click.option("-fix-li-amp/-fit-li-amp", default=None)
@click.option("--li-amp", type=float, default=0.0, callback=abs_cb)
@click.option("-fix-drag/-fit-drag", default=None)
@click.option("--drag", type=float, default=0.0, callback=abs_cb)
@click.option("--options-json", type=click.File("rb"), callback=readjson)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.argument(
    "filenames",
    nargs=-1,
    type=click.Path(
        exists=True,
        file_okay=True,
        dir_okay=False,
        readable=True,
        path_type=pathlib.Path,
    ),
    required=True,
    callback=suffix,
)
def main(
    fit_mode,
    output_path,
    k,
    defl_sens,
    sync_dist,
    trace,
    radius,
    fix_radius,
    tau,
    fix_tau,
    lj_scale,
    fix_lj_scale,
    vd,
    fix_vd,
    li_per,
    fix_li_per,
    li_amp,
    fix_li_amp,
    drag,
    fix_drag,
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
            (filename.parent / output_path).mkdir(parents=True, exist_ok=True)

    # convert flags to fitfix
    fit_fix = FitFix.DEFAULTS

    # override default flags with configuration file
    if options_json and "fit_fix" in options_json:
        fit_fix = options_json["fit_fix"]

    # override default and file flags with selections from command line
    for m in FitFix.__members__:
        this_flag = locals()["fix_" + m.lower()]
        if this_flag is not None:
            fit_fix &= ~FitFix[m]
            if this_flag:
                fit_fix |= FitFix[m]

    click.echo(
        (
            fit_mode,
            output_path,
            k,
            defl_sens,
            sync_dist,
            trace,
            radius,
            fix_radius,
            tau,
            fix_tau,
            lj_scale,
            fix_lj_scale,
            vd,
            fix_vd,
            li_per,
            fix_li_per,
            li_amp,
            fix_li_amp,
            drag,
            fix_drag,
            options_json,
            filenames,
        )
    )
