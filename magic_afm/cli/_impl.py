import enum
import pathlib

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

import click
import numpy as np
from attr import evolve
from tqdm import tqdm


from ..data_readers import SUFFIX_FVFILE_MAP, FVFile
from ..calculation import FitMode, FitFix, process_force_curve, calc_properties_imap


class TraceChoice(enum.IntEnum):
    RETRACE = 0
    TRACE = 1
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


# TODO: nice workers
ppe = ProcessPoolExecutor()


def threaded_opener(filenames):
    """Yield the opened fvfiles and names while opening the next in a background thread.

    This generator goes out of its way to avoid opening more than 2 files at a time."""
    first = True
    tpe = ThreadPoolExecutor()
    fvfile: FVFile
    for filename in filenames:
        if first:
            fvfile_cls, opener = SUFFIX_FVFILE_MAP[filename.suffix.lower()]
            fut = tpe.submit(opener, filename)
            prev_filename = filename
            first = False
            continue

        with fut.result() as open_thing:
            fvfile = fvfile_cls.parse(open_thing)
            fvfile_cls, opener = SUFFIX_FVFILE_MAP[filename.suffix.lower()]
            fut = tpe.submit(opener, filename)
            yield fvfile, prev_filename
            prev_filename = filename

    with fut.result() as open_thing:
        fvfile = fvfile_cls.parse(open_thing)
        yield fvfile, prev_filename


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
@click.option("--M", "M", type=float, default=1e9, callback=abs_cb)
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
    M,
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
    filenames: list[pathlib.Path],
):
    """test docstring

    more docstring"""

    # XXX: make ignoring numpy errors optional?
    np.seterr(all="ignore")

    # # prepare output folders early
    # if output_path is None:
    #     pass
    # elif output_path.is_absolute():
    #     output_path.mkdir(parents=True, exist_ok=True)
    # else:
    #     for filename in filenames:
    #         (filename.parent / output_path).mkdir(parents=True, exist_ok=True)

    # convert flags to fitfix
    fit_fix = FitFix.DEFAULTS

    # override default flags with configuration file
    if options_json and "fit_fix" in options_json:
        fit_fix = options_json["fit_fix"]

    # override default and file flags with selections from command line
    for m in FitFix.__members__:
        if m == "DEFAULTS":
            continue
        this_flag = locals()["fix_" + m.lower()]
        if this_flag is not None:
            fit_fix &= ~FitFix[m]
            if this_flag:
                fit_fix |= FitFix[m]

    for fvfile, filename in tqdm(
        threaded_opener(filenames),
        total=len(filenames),
        smoothing=0,
        miniters=1,
        leave=True,
        position=1,
        desc="Files completed",
        unit="file",
    ):
        with tqdm(
            desc=f"Preparing to fit {filename.name} force curves",
            smoothing=0.01,
            # smoothing_time=1,
            unit=" fits",
            bar_format="{desc} {bar} {elapsed}",
            leave=True,
            position=0,
        ) as pbar:
            prepare_flag = True
            k = fvfile.k if k is None else k
            v = fvfile.volumes[not trace]
            if sync_dist != getattr(fvfile, "sync_dist", None):
                v = evolve(v, sync_dist=sync_dist)
            rows, cols = fvfile.volumes[0].shape
            ncurves = rows * cols
            concurrent_submissions = []
            # TODO: Parallel
            for rc_zd in v.iter_curves():
                if prepare_flag:
                    prepare_flag = False
                    pbar.set_description_str(f"Fitting {filename.name} force curves")
                    pbar.reset(total=ncurves)
                    pbar.bar_format = None
                if fit_mode != FitMode.SKIP:
                    z_d_s_rc = process_force_curve(
                        rc_zd,
                        fit_mode,
                        (1.0 if defl_sens is None else defl_sens / fvfile.defl_sens),
                    )
                    calc_properties_imap(
                        z_d_s_rc,
                        k=k,
                        radius=radius,
                        M=M,
                        tau=tau,
                        lj_scale=lj_scale,
                        vd=vd,
                        li_per=li_per,
                        li_amp=li_amp,
                        drag=drag,
                        fit_fix=fit_fix,
                        fit_mode=fit_mode,
                    )
                    # TODO: output
                pbar.update()
