import enum
import json
import pathlib
import sys

from multiprocessing import get_context
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, wait
from functools import partial
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
    PARMS_UNITS_DICT,
    PARMS_DTYPE,
)
from magic_afm._util import cli_init, MAX_WORKERS


class TraceChoice(enum.IntEnum):
    RETRACE = 0
    TRACE = 1
    # BOTH = 2
    # ALL = -1


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

OPTIONS_JSON_SCHEMA = dict(
    k=float,
    defl_sens=float,
    sync_dist=float,
    trace=TraceChoice.__getitem__,
    radius=float,
    M=float,
    tau=float,
    lj_scale=float,
    vd=float,
    li_per=float,
    li_amp=float,
    drag=float,
    fit_fix=FitFix,
    fit_mode=FitMode.__getitem__,
)

NULLABLE_FIELDS = {"k", "defl_sens", "sync_dist", "trace"}


def echo(message=None, file=None, nl=True, err=False, color=None):
    with tqdm.external_write_mode(file=sys.stderr if err else sys.stdout):
        click.echo(message, file, nl, err, color)


def abs_cb(c, p, v):
    return abs(v)


def readjson(c, p, options_json):
    if options_json is not None:
        import json

        with options_json:
            options_json = json.load(options_json)
        for k, value in list(options_json.items()):
            if k in NULLABLE_FIELDS and value is None:
                continue
            validator = OPTIONS_JSON_SCHEMA[k]
            options_json[k] = validator(value)
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
            unknown.append(filename.name)
    if unknown:
        raise ValueError(f"Unknown filetypes for {unknown}")
    return filenames


def _chunk_producer(fn, job_items, chunksize):
    while x := tuple(islice(job_items, chunksize)):
        yield fn, x


def _chunk_consumer(chunk):
    return tuple(map(*chunk))


def wait_and_process(
    concurrent_submissions, property_map, parms_map, parms_err_map, nan_counter, pbar
):
    done, concurrent_submissions = wait(
        concurrent_submissions, return_when="FIRST_COMPLETED"
    )
    for fut in done:
        for rc, properties, parms, parms_err in fut.result():
            if properties is None:
                property_map[rc] = np.nan
                parms_map[rc] = np.nan
                parms_err_map[rc] = np.nan
                nan_count = next(nan_counter)
            else:
                property_map[rc] = properties
                parms_map[rc] = parms
                parms_err_map[rc] = parms_err
                nan_count = None
            if pbar.update() and nan_count is not None:
                pbar.set_postfix(bad_fits=nan_count)

    return concurrent_submissions


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


@cloup.command(epilog="See https://github.com/richardsheridan/magic-afm for more.")
@cloup.option_group(
    "Mode selection",
    click.option(
        "--fit-mode",
        type=click.Choice(FitMode, case_sensitive=False),
        default=FitMode.RETRACT,
    ),
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
    click.option("-fix-radius/-fit-radius"),
    click.option("--radius", type=float, callback=abs_cb, default=20.0),
    # click.option("-fix-M/-fit-M"), # TODO: implement with constraint
    click.option("--M", "M", type=float, callback=abs_cb, default=1e9),
    click.option("-fix-tau/-fit-tau"),
    click.option("--tau", type=float, callback=clip(0.0, 1.0), default=0.0),
    click.option("-fix-lj-scale/-fit-lj-scale"),
    click.option("--lj-scale", type=float, callback=clip(-6.0, 6.0), default=2.0),
    click.option("-fix-vd/-fit-vd"),
    click.option("--vd", type=float, default=0.0),
    click.option("-fix-li-per/-fit-li-per"),
    click.option("--li-per", type=float, callback=abs_cb, default=0.0),
    click.option("-fix-li-amp/-fit-li-amp"),
    click.option("--li-amp", type=float, callback=abs_cb, default=0.0),
    click.option("-fix-drag/-fit-drag"),
    click.option("--drag", type=float, callback=abs_cb, default=0.0),
)
@click.option("--options-json", type=click.File("rb"), callback=readjson)
@click.option(
    "--output-path",
    type=click.Path(file_okay=False, dir_okay=True, path_type=pathlib.Path),
)
@click.option("--output-type", type=click.Choice(EXPORTER_MAP), default="npy")
@click.option("--verbose", is_flag=True)
@click.option("--disable-progress", is_flag=True)
@click.option("--stop-on-error", is_flag=True)
@click.option("--jit", is_flag=True)
@click.option("--k-sens/--no-k-sens", is_flag=True, default=True)
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
    options_json,
    output_path,
    output_type,
    verbose,
    disable_progress,
    stop_on_error,
    jit,
    filenames: list[pathlib.Path],
    **calculation_options,
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
    # short name
    # TODO: validate against (nonexistent) single source of truth in calculation module
    co = calculation_options

    def is_default(
        commandline_option_key,
        _get_source=click.get_current_context().get_parameter_source,
        _default_sources=(
            click.core.ParameterSource.DEFAULT,
            click.core.ParameterSource.DEFAULT_MAP,
        ),
    ):
        return _get_source(commandline_option_key) in _default_sources

    # get flag defaults from single source of truth, not click decorators
    co["fit_fix"] = FitFix.DEFAULTS

    # override use configuration file for defaults if given, but give command line
    # selections priority over the json file.
    if options_json:
        for k, v in options_json.items():
            if k == "fit_fix" or is_default(k):
                co[k] = v

    # override fit_fix with flags from command line
    for m in FitFix.__members__:
        if m == "DEFAULTS":
            continue
        this_flag = "fix_" + m.lower()
        # the "defaults" of the click decorators are totally ignored
        if not is_default(this_flag):
            # first unconditionally clear the flag
            co["fit_fix"] &= ~FitFix[m]
            # then conditionally set the flag
            if co[this_flag]:
                co["fit_fix"] |= FitFix[m]

    # prepare output folders early
    if co["fit_mode"] == FitMode.SKIP:
        if verbose:
            echo("Skipping creating output folders")
    elif output_path is None:
        pass
    elif output_path.is_absolute():
        if verbose:
            echo("Creating " + str(output_path))
        output_path.mkdir(parents=True, exist_ok=True)
    else:
        to_create = set()
        for filename in filenames:
            to_create.add(filename.parent / output_path)
        for filename in to_create:
            if verbose:
                echo("Creating " + str(filename))
            filename.mkdir(parents=True, exist_ok=True)
        del to_create

    ppe = ProcessPoolExecutor(
        # respect undocumented platform limit
        MAX_WORKERS if sys.platform != "win32" else min(MAX_WORKERS, 63 - 2),
        initializer=partial(cli_init, jit),
        mp_context=get_context("spawn"),
    )

    for fvfile, filename in tqdm(
        threaded_opener(filenames),
        total=len(filenames),
        smoothing=0,
        miniters=1,
        leave=True,
        position=0,
        desc="Files completed",
        unit="file",
        disable=disable_progress,
    ):
        try:
            with tqdm(
                desc=f"Preparing to fit {filename.name} force curves",
                smoothing=0.01,
                # smoothing_time=1,
                unit=" fits",
                bar_format="{desc} {bar} {elapsed}",
                leave=True,
                position=1,
                disable=disable_progress,
            ) as pbar:
                prepare_flag = True

                # per file defaults overridden here
                k = co["k"] or fvfile.k
                defl_sens = co["defl_sens"] or fvfile.defl_sens
                sync_dist = co["sync_dist"] or getattr(fvfile, "sync_dist", None)
                trace = co["trace"] is None or co["trace"]

                v = fvfile.volumes[not trace]
                if sync_dist != getattr(fvfile, "sync_dist", None):
                    v = evolve(v, sync_dist=sync_dist)

                imshape = v.shape
                property_map = np.empty(imshape, dtype=PROPERTY_DTYPE)
                parms_map = np.empty(imshape, dtype=PARMS_DTYPE)
                parms_err_map = np.empty(imshape, dtype=PARMS_DTYPE)
                concurrent_submissions = set()
                nan_counter = count(1)

                procfun = partial(
                    process_force_curve,
                    fit_mode=co["fit_mode"],
                    s_ratio=defl_sens / fvfile.defl_sens,
                )
                calcfun = partial(calc_properties_imap, **{**co, "k": k})
                chunksize = 8
                for rc_zd_chunk in _chunk_producer(procfun, v.iter_curves(), chunksize):
                    if prepare_flag:
                        prepare_flag = False
                        pbar.set_description_str(
                            f"Fitting {filename.name} force curves"
                        )
                        pbar.reset(total=imshape[0] * imshape[1])
                        pbar.bar_format = None

                    if co["fit_mode"] == FitMode.SKIP:
                        pbar.update(len(rc_zd_chunk[1]))
                        continue

                    z_d_s_rc_chunk = _chunk_consumer(rc_zd_chunk)
                    concurrent_submissions.add(
                        ppe.submit(
                            _chunk_consumer,
                            (calcfun, z_d_s_rc_chunk),
                        )
                    )

                    if len(concurrent_submissions) < MAX_WORKERS:
                        continue

                    concurrent_submissions = wait_and_process(
                        concurrent_submissions,
                        property_map,
                        parms_map,
                        parms_err_map,
                        nan_counter,
                        pbar,
                    )

                while concurrent_submissions:
                    concurrent_submissions = wait_and_process(
                        concurrent_submissions,
                        property_map,
                        parms_map,
                        parms_err_map,
                        nan_counter,
                        pbar,
                    )
        except Exception as e:
            message = f"Unhandled error processing {str(filename)}."
            if stop_on_error:
                raise RuntimeError(message) from e
            else:
                echo(message + " Continuing...", err=True)
                continue

        if co["fit_mode"] == FitMode.SKIP:
            if verbose:
                echo("Skipping writing options_json and property maps")
            continue

        # Actually write out results to external world
        if is_default("trace"):
            tracestr = ""
        elif trace == 0:
            tracestr = "Retrace"
        elif trace == 1:
            tracestr = "Trace"
        else:
            tracestr = str(trace)
        if co["fit_mode"] == FitMode.EXTEND:
            extret = "Ext"
        elif co["fit_mode"] == FitMode.RETRACT:
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

        # recreate options_json dict with selections
        new_json = {k: co[k] for k in OPTIONS_JSON_SCHEMA}
        # save per-file defaults/cli overrides
        new_json["k"] = k
        new_json["defl_sens"] = defl_sens
        new_json["sync_dist"] = sync_dist
        # convert enums to names
        # TODO: convert via single source of truth
        new_json["fit_mode"] = FitMode(new_json["fit_mode"]).name
        if new_json["trace"] is not None:
            new_json["trace"] = TraceChoice(co["trace"]).name

        if verbose:
            echo("Writing " + str(options_json_path))
        with options_json_path.open("w") as fp:
            json.dump(new_json, fp)

        for names, map_, err_str in zip(
            [PROPERTY_UNITS_DICT, PARMS_UNITS_DICT, PARMS_UNITS_DICT],
            [property_map, parms_map, parms_err_map],
            ["", "", "Err"],
        ):
            for name in names:
                units_factor = 1  # would be great to use UNITS_DICT somehow
                name2 = name
                # skip writing fittable params that are fixed
                if (
                    name.upper() in FitFix.__members__
                    and co["fit_fix"] & FitFix[name.upper()]
                ):
                    continue
                # ugly special case for M, etc.
                if name == "M":
                    if not co["fit_fix"] & FitFix["RADIUS"]:
                        continue
                    name2 = "IndentationModulus"
                    units_factor = 1e9
                if name == "li_pha":
                    if co["fit_fix"] & FitFix["LI_AMP"]:
                        continue
                if name == "fc":
                    name2 = "AdhesionForce"
                    units_factor = 1e-9
                if name == "SensIndMod_k" and not co["k_sens"]:
                    continue
                export_path = parent_path / (
                    filename.stem
                    + "_"
                    + extret
                    + name2
                    + err_str
                    + tracestr
                    + "."
                    + output_type
                )
                if verbose:
                    echo("Writing " + str(export_path))
                EXPORTER_MAP[output_type](
                    export_path, units_factor * map_[name].squeeze()[::-1]
                )
