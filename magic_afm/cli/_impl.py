import enum
import json
import os
import pathlib

from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial, wraps
from itertools import islice, count

import click
import cloup
import imageio
import numpy as np
from attr import evolve
from tqdm import tqdm


from magic_afm.data_readers import SUFFIX_FVFILE_MAP, FVFile
from magic_afm.calculation import (
    FitMode,
    FitFix,
    process_force_curve,
    calc_properties_imap,
    PROPERTY_UNITS_DICT,
    PROPERTY_DTYPE,
)
from magic_afm._util import nice_workers


class TraceChoice(enum.IntEnum):
    RETRACE = 0
    TRACE = 1
    # BOTH = 2
    # ALL = -1


def pass_none(f):
    @wraps(f)
    def pass_none_inner(c, p, v):
        if v is None:
            return None
        return f(c, p, v)

    return pass_none_inner


@pass_none
def abs_cb(c, p, v):
    return abs(v)


def readjson(c, p, options_json):
    if options_json is not None:
        import json

        with options_json:
            options_json = json.load(options_json)
    return options_json


def clip(lo, hi):
    @pass_none
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


def _chunk_producer(fn, job_items, chunksize):
    while x := tuple(islice(job_items, chunksize)):
        yield fn, x


def _chunk_consumer(chunk):
    return tuple(map(*chunk))


def wait_and_process(concurrent_submissions, property_map, nan_counter, pbar):
    done, concurrent_submissions = wait(
        concurrent_submissions, return_when="FIRST_COMPLETED"
    )
    for fut in done:
        for rc, properties in fut.result():
            if properties is None:
                property_map[rc] = np.nan
                nan_count = next(nan_counter)
            else:
                property_map[rc] = properties
                nan_count = None
            if pbar.update() and nan_count is not None:
                pbar.set_postfix(bad_fits=nan_count)

    return concurrent_submissions


# must take two positional arguments, fname and array
EXPORTER_MAP = {
    "txt": partial(np.savetxt, fmt="%.8g"),
    "asc": partial(np.savetxt, fmt="%.8g"),
    "tsv": partial(np.savetxt, fmt="%.8g", delimiter="\t"),
    "csv": partial(np.savetxt, fmt="%.8g", delimiter=","),
    "tif": imageio.imwrite,
    "npy": np.save,
    "npz": np.savez_compressed,
}


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


@cloup.command(epilog="See https://github.com/richardsheridan/magic-afm for details.")
@cloup.option_group(
    "Mode selection",
    click.option("--fit-mode", type=click.Choice(FitMode, case_sensitive=False)),
    click.option("--trace", type=click.Choice(TraceChoice, case_sensitive=False)),
)
@cloup.option_group(
    "Data parameters",
    click.option("--k", type=float),
    click.option("--defl-sens", type=float),
    click.option("--sync-dist", type=float),
)
@cloup.option_group(
    "Fit parameters",
    click.option("-fix-radius/-fit-radius", default=None),
    click.option("--radius", type=float, callback=abs_cb),
    # click.option("-fix-M/-fit-M", default=None), # TODO: implement with constraint
    click.option("--M", "M", type=float, callback=abs_cb),
    click.option("-fix-tau/-fit-tau", default=None),
    click.option("--tau", type=float, callback=clip(0.0, 1.0)),
    click.option("-fix-lj-scale/-fit-lj-scale", default=None),
    click.option("--lj-scale", type=float, callback=clip(-6.0, 6.0)),
    click.option("-fix-vd/-fit-vd", default=None),
    click.option("--vd", type=float),
    click.option("-fix-li-per/-fit-li-per", default=None),
    click.option("--li-per", type=float, callback=abs_cb),
    click.option("-fix-li-amp/-fit-li-amp", default=None),
    click.option("--li-amp", type=float, callback=abs_cb),
    click.option("-fix-drag/-fit-drag", default=None),
    click.option("--drag", type=float, callback=abs_cb),
)
@click.option("--options-json", type=click.File("rb"), callback=readjson)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--output-type", type=click.Choice(EXPORTER_MAP), default="npy")
@click.option("--verbose", is_flag=True)
@click.option("--disable-progress", is_flag=True)
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
    output_path,
    output_type,
    verbose,
    disable_progress,
    filenames: list[pathlib.Path],
):
    """Fit all force curves in FILENAMES with the MagicAFM LJ/SCHWARZ model.

    By default, the fits will use the retract curve and sane fit initializations
    and constraints, and output the results in npy format to the same directory
    as the input file.

    These can be overridden by specifying an options-json, which can itself be
    overridden by fix/fit switches and initial value options at the command line.
    Note that M (modulus) and radius are covariate so their fit/fix flag is linked.

    If output-path is "absolute" (i.e. starts with "C:\\" or "/"), all results
    will be written to the same directory, and identically-named files will clobber
    each other.

    If output-path is "relative", then a directory will be created in the same
    directory as the input file, and identically-named files in the same directory
    will clobber each other.
    """

    exporter = EXPORTER_MAP[output_type]

    # convert flags to fitfix
    fit_fix = FitFix.DEFAULTS

    # override use configuration file for defaults if given, else default defaults
    if options_json:
        fit_fix = options_json["fit_fix"]
        k = options_json["k"] if k is None else k
        defl_sens = options_json["defl_sens"] if defl_sens is None else defl_sens
        sync_dist = options_json["sync_dist"] if sync_dist is None else sync_dist
        trace = options_json["trace"] if trace is None else trace
        radius = options_json["radius"] if radius is None else radius
        M = options_json["M"] if M is None else M
        tau = options_json["tau"] if tau is None else tau
        lj_scale = options_json["lj_scale"] if lj_scale is None else lj_scale
        vd = options_json["vd"] if vd is None else vd
        li_per = options_json["li_per"] if li_per is None else li_per
        li_amp = options_json["li_amp"] if li_amp is None else li_amp
        drag = options_json["drag"] if drag is None else drag
        fit_mode = options_json["fit_mode"] if fit_mode is None else fit_mode
    else:
        # handle trace locally
        # trace = TraceChoice.TRACE if trace is None else trace
        radius = 20.0 if radius is None else radius
        M = 1e9 if M is None else M
        tau = 0.0 if tau is None else tau
        lj_scale = 2.0 if lj_scale is None else lj_scale
        vd = 0.0 if vd is None else vd
        li_per = 0.0 if li_per is None else li_per
        li_amp = 0.0 if li_amp is None else li_amp
        drag = 0.0 if drag is None else drag
        fit_mode = FitMode.RETRACT if fit_mode is None else fit_mode

    # override default and file flags with selections from command line
    for m in FitFix.__members__:
        if m == "DEFAULTS":
            continue
        this_flag = locals()["fix_" + m.lower()]
        if this_flag is not None:
            fit_fix &= ~FitFix[m]
            if this_flag:
                fit_fix |= FitFix[m]

    # prepare output folders early
    if fit_mode == FitMode.SKIP:
        if verbose:
            click.echo("Skipping creating output folders")
    elif output_path is None:
        pass
    elif output_path.is_absolute():
        if verbose:
            click.echo("Creating " + str(output_path))
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        to_create = set()
        for filename in filenames:
            to_create.add(filename.parent / output_path)
        for filename in to_create:
            if verbose:
                click.echo("Creating " + str(filename))
            filename.mkdir(parents=True, exist_ok=True)

    # if needed, create options_json dict
    if not options_json:
        options_json = dict(
            k=k,
            defl_sens=defl_sens,
            sync_dist=sync_dist,
            trace=trace,
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

    max_workers = os.process_cpu_count() or 1
    ppe = ProcessPoolExecutor(max_workers, initializer=nice_workers)

    for fvfile, filename in tqdm(
        threaded_opener(filenames),
        total=len(filenames),
        smoothing=0,
        miniters=1,
        leave=True,
        position=1,
        desc="Files completed",
        unit="file",
        disable=disable_progress,
    ):
        with tqdm(
            desc=f"Preparing to fit {filename.name} force curves",
            smoothing=0.01,
            # smoothing_time=1,
            unit=" fits",
            bar_format="{desc} {bar} {elapsed}",
            leave=True,
            position=0,
            disable=disable_progress,
        ) as pbar:
            prepare_flag = True
            k = fvfile.k if k is None else k
            v = fvfile.volumes[not (trace is None or trace)]
            if sync_dist != getattr(fvfile, "sync_dist", None):
                v = evolve(v, sync_dist=sync_dist)
            rows, cols = v.shape
            ncurves = rows * cols
            property_map = np.empty((rows, cols), dtype=PROPERTY_DTYPE)
            concurrent_submissions = set()
            nan_counter = count(1)
            procfun = partial(
                process_force_curve,
                fit_mode=fit_mode,
                s_ratio=(1.0 if defl_sens is None else defl_sens / fvfile.defl_sens),
            )
            calcfun = partial(
                calc_properties_imap,
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
            chunksize = 8
            for rc_zd_chunk in _chunk_producer(procfun, v.iter_curves(), chunksize):
                if prepare_flag:
                    prepare_flag = False
                    pbar.set_description_str(f"Fitting {filename.name} force curves")
                    pbar.reset(total=ncurves)
                    pbar.bar_format = None

                if fit_mode == FitMode.SKIP:
                    pbar.update(len(rc_zd_chunk[1]))
                    continue

                z_d_s_rc_chunk = _chunk_consumer(rc_zd_chunk)
                concurrent_submissions.add(
                    ppe.submit(
                        _chunk_consumer,
                        (calcfun, z_d_s_rc_chunk),
                    )
                )

                if len(concurrent_submissions) < max_workers:
                    continue

                concurrent_submissions = wait_and_process(
                    concurrent_submissions, property_map, nan_counter, pbar
                )

            while concurrent_submissions:
                concurrent_submissions = wait_and_process(
                    concurrent_submissions, property_map, nan_counter, pbar
                )

        if fit_mode == FitMode.SKIP:
            if verbose:
                click.echo("Skipping writing options_json and property maps")
            continue

        # Actually write out results to external world
        if trace == 0:
            tracestr = "Retrace"
        elif trace == 1:
            tracestr = "Trace"
        else:
            tracestr = ""
        if fit_mode == FitMode.EXTEND:
            extret = "Ext"
        elif fit_mode == FitMode.RETRACT:
            extret = "Ret"
        else:
            extret = "Both"

        if output_path is None:
            parent_path = filename.parent
        elif output_path.is_absolute():
            parent_path = output_path
        else:
            parent_path = filename.parent / output_path

        options_json_path = parent_path / (
            filename.stem + "_options" + extret + tracestr + ".json"
        )
        if verbose:
            click.echo("Writing " + str(options_json_path))
        with options_json_path.open("w") as fp:
            json.dump(options_json, fp)

        for name in PROPERTY_UNITS_DICT:
            export_path = parent_path / (
                filename.stem + "_" + extret + name + tracestr + "." + output_type
            )
            if verbose:
                click.echo("Writing " + str(export_path))
            exporter(export_path, property_map[name].squeeze()[::-1])
