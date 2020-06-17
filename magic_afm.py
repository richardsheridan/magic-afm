from functools import partial

import h5py
import numpy as np
import subprocess
import trio
from scipy.optimize import curve_fit, root_scalar
from scipy.signal import resample


trs = partial(trio.to_thread.run_sync, cancellable=True)

EPS = float(np.finfo(np.float64).eps)
RT_EPS = float(np.sqrt(EPS))

def make_cancel_poller():
    """Uses internal undocumented bits so probably super fragile"""
    _cancel_status=trio.lowlevel.current_task()._cancel_status
    def poll_for_cancel():
        if _cancel_status.effectively_cancelled:
            raise trio.Cancelled._create()
    return poll_for_cancel

async def convert_ardf(ardf_path, conv_path=r'X:\Data\AFM\Cypher\ARDFtoHDF5.exe', force=False, pbar=None):
    """Turn an ARDF path into a corresponding HDF5 path, converting the file if it doesn't exist.
    
    Can force the conversion with the force flag if necessary (e.g. overwriting with new data).
    Requires converter executable available from Asylum Research"""
    ardf_path = trio.Path(ardf_path)
    #     conv_path = trio.Path(conv_path)
    h5file_path = ardf_path.with_suffix('.h5')

    if (not force) and (await h5file_path.is_file()):
        return h5file_path

    if pbar is None:
        import tqdm
        pbar = tqdm.tqdm(total=100, unit='%')

    async def reading_stdout():
        stdout = []
        async for stuff in proc.stdout:
            stdout.append(stuff)
        stdout = b''.join(stdout).decode()
        print(stdout)
        return stdout

    async def reading_stderr():
        async for stuff in proc.stderr:
            stuff = (stuff.lstrip(b'\x08'))
            if stuff:
                pbar.update(float(stuff[:-1].decode()) - pbar.n)
        pbar.close()

    try:
        async with await trio.open_process([str(conv_path), str(ardf_path), str(h5file_path), ],
                                           stderr=subprocess.PIPE, stdout=subprocess.PIPE,
                                           ) as proc:
            async with trio.open_nursery() as nursery:
                nursery.start_soon(reading_stdout)
                nursery.start_soon(reading_stderr)
    except FileNotFoundError as e:
        print('Please ensure the full path to the Asylum converter tool ARDFtoHDF5.exe is in the conv_path argument',
              flush=True)
        raise e

    return h5file_path


class ForceMapWorker:
    def __init__(self, h5data):
        self.force_curves = h5data['ForceMap']['0']
        # ForceMap Segments can contain 3 or 4 endpoint indices for each indent array
        self.segments = self.force_curves['Segments'][:, :, :]  # XXX Read h5data
        im_r, im_c, num_segments = self.segments.shape

        # Generally segments are [Ext, Dwell, Ret, Away] or [Ext, Ret, Away]
        # for magic, we don't dwell. new converter ensures this assertion
        assert num_segments == 3

        # this is all necessary because the arrays are not of uniform length
        # We will cut all arrays down to the length of the smallest
        self.extlens = self.segments[:, :, 0]
        self.minext = np.min(self.extlens)
        self.extretlens = self.segments[:, :, 1]
        self.minret = np.min(self.extretlens - self.extlens)

        # We only care about 2 channels, Defl and ZSnsr
        # Convert channels array to a map that can be used to index into ForceMap data by name
        # chanmap should always be {'Defl':1,'ZSnsr':2} but it's cheap to calculate
        self.chanmap = {key.decode('utf8'): index for index, key in enumerate(self.force_curves.attrs['Channels'])}

    def _shared_get_part(self, curve, s):
        # Index into the data and grab the Defl and Zsnsr ext and ret arrays as one 2D array
        # We could slice with "1:" if chanmap were constant but I'm not sure if it is
        defl_zsnsr_rows = [self.chanmap['Defl'], self.chanmap['ZSnsr']]
        defl_zsnsr = curve[defl_zsnsr_rows, :]  # XXX Read h5data

        # we are happy to throw away data far from the surface to square up the data
        # Also reverse axis zero so data is ordered zsnsr,defl like we did for FFM
        return defl_zsnsr[::-1, (s - self.minext):(s + self.minret)]

    def get_force_curve(self, r, c):
        # Because of the nonuniform arrays, each indent gets its own dataset
        # indexed by 'row:column' e.g. '1:1'.
        curve = self.force_curves[f'{r}:{c}']  # XXX Read h5data
        s = self.extlens[r, c]

        return self._shared_get_part(curve, s)

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        im_r, im_c, num_segments = self.segments.shape
        x = np.empty((im_r, im_c, 2, self.minext + self.minret), dtype=np.float32)
        for index, curve in self.force_curves.items():
            # Unfortunately they threw in segments here too, so we skip over it
            if index == 'Segments':
                continue
            _poll_for_cancel()
            # Because of the nonuniform arrays, each indent gets its own dataset
            # indexed by 'row:column' e.g. '1:1'. We could start with the shape and index
            # manually, but the string munging is easier for me to think about
            r, c = index.split(':')
            r, c = int(r), int(c)
            s = self.extlens[r, c]

            x[r, c, :, :] = self._shared_get_part(curve, s)
        return x


class FFMSingleWorker:
    def __init__(self, drive, defl):
        self.drive = drive
        self.defl = defl

    def get_force_curve(self, r, c):
        z = self.drive[r, c]
        d = self.defl[r, c]
        return z, d

    def get_all_curves(self, _poll_for_cancel=None):
        return np.stack((self.drive, self.defl), axis=-2)


class FFMTraceRetraceWorker:
    def __init__(self, drive_trace, defl_trace, drive_retrace, defl_retrace):
        self.drive_trace = drive_trace
        self.defl_trace = defl_trace
        self.drive_retrace = drive_retrace
        self.defl_retrace = defl_retrace
        self.trace = True

    def get_force_curve(self, r, c):
        if self.trace:
            z = self.drive_trace[r, c]
            d = self.defl_trace[r, c]
        else:
            z = self.drive_retrace[r, c]
            d = self.defl_retrace[r, c]
        return z, d

    def get_all_curves(self, _poll_for_cancel=(lambda: None)):
        drive = np.concatenate((self.drive_trace, self.drive_retrace))
        _poll_for_cancel()
        defl = np.concatenate((self.defl_trace, self.defl_retrace))
        _poll_for_cancel()
        return np.stack((drive, defl,), axis=-2)


class AsyncARH5File:
    def __init__(self, h5file_path):
        self.h5file_path = h5file_path
        self._h5data = None
        self.notes = None
        self._worker = None

    async def ainitialize(self):
        h5data = await trs(h5py.File, self.h5file_path, "r")
        # The notes have a very regular key-value structure, so we convert to dict for later access
        notes = await trs(
            lambda: dict(line.split(':', 1)
                         for line in h5data.attrs["Note"].decode('utf8').split('\n')
                         if ':' in line))
        worker = await trs(self._choose_worker, h5data)
        self._h5data = h5data
        self.notes = notes
        self._worker = worker

    async def aclose(self):
        await trs(self._h5data.close, cancellable=False)

    async def __aenter__(self):
        await self.ainitialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.aclose()

    @staticmethod
    def _choose_worker(h5data):
        if 'FFM' in h5data:
            if '1' in h5data['FFM']:
                worker = FFMTraceRetraceWorker(h5data["FFM"]["0"]["Drive"],
                                               h5data["FFM"]["0"]["Defl"],
                                               h5data["FFM"]["1"]["Drive"],
                                               h5data["FFM"]["1"]["Defl"],
                                               )
            elif '0' in h5data["FFM"]:
                worker = FFMSingleWorker(h5data["FFM"]["0"]["Drive"],
                                         h5data["FFM"]["0"]["Defl"],
                                         )
            else:
                worker = FFMSingleWorker(h5data["FFM"]["Drive"],
                                         h5data["FFM"]["Defl"],
                                         )
        else:
            worker = ForceMapWorker(h5data)
        return worker

    async def get_force_curve(self, r, c):
        return await trs(self._worker.get_force_curve, r, c)

    async def get_all_curves(self):
        return await trs(self._worker.get_all_curves, make_cancel_poller())


def parse_notes(notes, disp=True):
    """Extract k, force_setpoint, z_rate, fs values from notes dict values"""
    if int(notes['ForceMapImage']):
        k = float(notes['SpringConstant'])
        force_setpoint = float(notes['TriggerPoint']) * 1e9
        z_rate = float(notes['ForceScanRate'])
        fs = float(notes['NumPtsPerSec'])
    elif int(notes['FastMapImage']):
        k = float(notes['SpringConstant'])
        force_setpoint = float(notes['FastMapSetpointNewtons']) * 1e9
        z_rate = float(notes['FastMapZRate'])
        fs = float(notes['NumPtsPerSec'])
    else:
        raise ValueError('Cannot identify data type: Neither ForceMap nor Fastmap')

    if disp:
        print(notes['ImageNote'])
        print('k =', k, 'N/m')
        print('F =', force_setpoint, 'nN')
        print('rate =', z_rate, 'Hz')
        print('SamplingFreq fs =', fs, 'Hz')
    return k, force_setpoint, z_rate, fs


def resample_dset(X, npts, fourier):
    """Consistent API for resampling force curves with Fourier or interpolation"""
    X = np.atleast_2d(X)
    if fourier:
        return resample(X, npts, axis=1, window=('kaiser', 6), ).squeeze()
    else:
        tnew = np.linspace(0, 1, npts, endpoint=False)
        told = np.linspace(0, 1, X.shape[-1], endpoint=False)
        return np.stack([np.interp(tnew, told, x, ) for x in X]).squeeze()


def secant(func, x0, x1=None):
    """Secant method from scipy optimize but stripping np.isclose for speed
    
Copyright (c) 2001-2002 Enthought, Inc.  2003-2019, SciPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions
are met:

1. Redistributions of source code must retain the above copyright
   notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above
   copyright notice, this list of conditions and the following
   disclaimer in the documentation and/or other materials provided
   with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived
   from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
    p0 = x0 * 1.0
    tol = RT_EPS
    maxiter = 50
    # rtol=0.0
    # funcalls = 0
    if x1 is not None:
        if x1 == x0:
            raise ValueError("x1 and x0 must be different")
        p1 = x1 * 1.0
    else:
        eps = RT_EPS
        p1 = x0 * (1 + eps)
        p1 += (eps if p1 >= 0 else -eps)
    q0 = func(p0, )
    # funcalls += 1
    q1 = func(p1, )
    # funcalls += 1
    if abs(q1) < abs(q0):
        p0, p1, q0, q1 = p1, p0, q1, q0
    for itr in range(maxiter):
        if q1 == q0:
            #             if p1 != p0:
            #                 msg = "Tolerance of %s reached." % (p1 - p0)
            #                 print(msg, itr)
            p = (p1 + p0) / 2.0
            break
        else:
            if abs(q1) > abs(q0):
                p = (-q0 / q1 * p1 + p0) / (1 - q0 / q1)
            else:
                p = (-q1 / q0 * p0 + p1) / (1 - q1 / q0)
        #         print(itr,p)
        if abs(p - p1) <= tol:  # + rtol*abs(p):
            break
        p0, q0 = p1, q1
        p1 = p
        q1 = func(p1, )
        # funcalls += 1
    else:
        p = (p1 + p0) / 2.0
    #         print("Maximum iterations reached")

    return p.real


def mylinspace(start, stop, num, endpoint=True):
    """np.linspace is surprisingly intensive, so trim the fat
    
Copyright (c) 2005-2020, NumPy Developers.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are
met:

    * Redistributions of source code must retain the above copyright
       notice, this list of conditions and the following disclaimer.

    * Redistributions in binary form must reproduce the above
       copyright notice, this list of conditions and the following
       disclaimer in the documentation and/or other materials provided
       with the distribution.

    * Neither the name of the NumPy Developers nor the names of any
       contributors may be used to endorse or promote products derived
       from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""
    delta = stop - start
    div = (num - 1) if endpoint else num

    y = np.arange(0.0, num, )
    y *= delta / div
    y += start
    if endpoint and num > 1:
        y[-1] = stop
    return y


def schwarz_red(red_f, red_fc, stab):
    """Calculate Schwarz potential indentation depth from force in reduced units.
    
    """
    # if red_f<red_fc, it is likely a small numerical error
    # this works for arrays and python scalars without weird logic
    df = abs(red_f - red_fc)

    # Save computations if pure DMT
    if red_fc == -2:
        return df ** (2 / 3)

    # trade a branch for a multiply. does it make sense? dunno!
    if stab:
        red_contact_radius = (((3 * red_fc + 6) ** (1 / 2) + (df) ** (1 / 2))) ** (2 / 3)
    else:
        red_contact_radius = (((3 * red_fc + 6) ** (1 / 2) - (df) ** (1 / 2))) ** (2 / 3)

    # red_contact_radius = (((3*red_fc+6)**(1/2)+stab*(df)**(1/2)))**(2/3) # stab in {1,-1}

    red_delta = red_contact_radius ** 2 - 4 * (red_contact_radius * (red_fc + 2) / 3) ** (1 / 2)

    return red_delta


# Exponents for LJ potential
bigpow = 9
lilpow = 3
powrat = (bigpow / lilpow)

### minimum value at delta=1
prefactor = (powrat) ** (1 / (bigpow - lilpow))
postfactor = 1 / (prefactor ** (-bigpow) - prefactor ** (-lilpow))  # such that lj(1)=1
lj_limit_factor = ((bigpow + 1) / (lilpow + 1)) ** (1 / (bigpow - lilpow))  # delta of minimum slope


### minimum slope at delta=1 (most negative->steepest)
# prefactor= (powrat*(bigpow+1)/(lilpow+1))**(1/(bigpow-lilpow))
# lj_limit_factor = 1

def lj_force(delta, delta_scale, force_scale, delta_offset, force_offset=0):
    """Calculate a leonard-Jones force curve.
    
    Prefactor scaled such that minimum slope is at delta=1/delta_scale"""
    nondim_position = (delta - delta_offset) / delta_scale
    attraction = (prefactor * nondim_position) ** (-lilpow)
    return postfactor * force_scale * (attraction ** powrat - attraction) - force_offset


def lj_gradient(delta, delta_scale, force_scale, delta_offset, force_offset=0):
    """Gradient of lj_force.
    
    Offset is useful for root finding (reduced spring constant matching)"""
    nondim_position = (delta - delta_offset) / delta_scale
    attraction = lilpow * prefactor ** (-lilpow) * nondim_position ** (-lilpow - 1)
    repulsion = bigpow * prefactor ** (-bigpow) * nondim_position ** (-bigpow - 1)
    return postfactor * force_scale * (attraction - repulsion) / delta_scale - force_offset


def red_extend(red_delta, red_fc, red_k, lj_delta_scale, ):
    """Calculate, in reduced units, an extent Schwarz curve with long range LJ potential and snap-off physics.
    
    MUCH LESS TESTED THAN RED_RETRACT
    
    Following the broad example of Lin (2007) and Johnson (1997), we combine the stable branch of the
    Schwarz potential with a long range Leonard-Jones potential via finite-k snap-off physics.
    
    To do this, we build two curves, one for each physical model. The LJ potential is clipped
    above where its slope equals the reduced lever spring constant and replaced with a tangent line.
    The Schwarz potential is clipped at the intersection point and replaced with negative infinities.
    Doing so makes the higher of the two curves at all points give the result.
    
    Or in pithy math form we want
    maximum(NINF⌢S, LJ⌢snap)
    """
    # We can take a shortcut if precisely DMT physics since the unstable branch vanishes
    #     not_DMT = not red_fc==-2

    # The indentation at the critical force must be calculated directly, no shortcuts
    red_deltac = schwarz_red(red_fc, red_fc, 1)

    # lj_force() outputs a curve with minimum at (lj_delta_scale,lj_force_scale)
    # offset needed to put minimum at red_deltac
    lj_delta_offset = red_deltac - lj_delta_scale
    lj_f = lj_force(red_delta, lj_delta_scale, red_fc, lj_delta_offset)
    lj_f = np.atleast_1d(lj_f)  # Later parts can choke on scalars

    try:
        # Try to find where LJ gradient == red_k
        # root_scalar is marginally faster with python scalars
        red_d_min = float(np.min(red_delta))
        args = (lj_delta_scale, red_fc, lj_delta_offset, red_k)
        bracket = (red_d_min, lj_limit_factor * lj_delta_scale + lj_delta_offset,)
        lj_end_pos = root_scalar(lj_gradient, args=args, bracket=bracket, ).root
    except ValueError as e:
        if str(e) != 'f(a) and f(b) must have different signs':
            raise
        # If root finding fails, put the end pos somewhere quasi-reasonable
        if lj_gradient(red_d_min, *args) <= 0:
            lj_end_pos = red_d_min
        else:
            lj_end_pos = lj_delta_scale + lj_delta_offset

    lj_end_force = lj_force(lj_end_pos, lj_delta_scale, red_fc, lj_delta_offset)

    # Overwrite after end position with tangent line
    lj_f[red_delta >= lj_end_pos] = (red_delta[red_delta >= lj_end_pos] - lj_end_pos) * red_k + lj_end_force

    # Unfortunately Schwarz is d(f) rather than f(d) so we need to infer the largest possible force    
    red_d_max = np.max(red_delta)
    red_fc, red_d_max = float(red_fc), float(red_d_max)
    red_f_max = secant(lambda x: schwarz_red(x, red_fc, 1) - red_d_max, x0=0, x1=red_fc)

    # because of noise in force channel, need points well beyond red_f_max
    # XXX: a total hack, would be nice to have a real stop and num for this linspace
    f = mylinspace(red_fc, 1.5 * (red_f_max - red_fc) + red_fc, 100)
    d = schwarz_red(f, red_fc, 1)

    # NOTE: only use stable branch for snap in, duh
    #     if not_DMT:
    #         f0 = mylinspace((7*red_fc+8)/3,red_fc,100,endpoint=False)
    #         d0 = schwarz_red(f0,red_fc,0)
    #         f = np.concatenate((f0,f))
    #         d = np.concatenate((d0,d))

    s_f = np.interp(red_delta, d, f, left=np.NINF)

    return np.maximum(s_f, lj_f)


def red_retract(red_delta, red_fc, red_k, lj_delta_scale):
    """Calculate, in reduced units, a retract Schwarz curve with long range LJ potential and snap-off physics.
    
    Following the broad example of Lin (2007) and Johnson (1997), we combine the unstable branch
    of the Schwarz potential with a long range Leonard-Jones potential via finite-k snap-off physics.
    
    To do this, we build two curves, one for each physical model. The unstable Schwarz branch is clipped
    below where its slope equals the reduced lever spring constant and replaced with a tangent line.
    The LJ potential is clipped at it's minimum (positioned by definition at (red_deltac, red_fc)) and
    replaced with infinities. Doing so makes the lower of the two curves at all points give the result.
    
    Or in pithy math form we want
    minimum(snap⌢S, LJ⌢INF)
    """
    # We can take a shortcut if precisely DMT physics since the unstable branch vanishes
    not_DMT = not red_fc == -2

    # The indentation at the critical force must be calculated directly, no shortcuts
    red_deltac = schwarz_red(red_fc, red_fc, 1)

    # lj_force() outputs a curve with minimum at (lj_delta_scale,lj_force_scale)
    # offset needed to put minimum at red_deltac
    lj_delta_offset = red_deltac - lj_delta_scale
    lj_f = lj_force(red_delta, lj_delta_scale, red_fc, lj_delta_offset)
    lj_f = np.atleast_1d(lj_f)  # Later parts can choke on scalars

    # make points definitely lose in minimum function if in the contact/stable branch
    #     lj_f[red_delta>=lj_limit_factor*lj_delta_scale+lj_delta_offset] = np.inf # use if LJ based on minimum slope
    lj_f[red_delta >= red_deltac] = np.inf  # use if LJ based on minimum value

    # Unfortunately Schwarz is d(f) rather than f(d) so we need to infer the largest possible force
    red_d_max = np.max(red_delta)
    red_fc, red_d_max = float(red_fc), float(red_d_max)
    red_f_max = secant(lambda x: schwarz_red(x, red_fc, 1) - red_d_max, x0=red_fc, x1=0)

    # because of noise in force channel, need points well beyond red_f_max
    # XXX: a total hack, would be nice to have a real stop and num for this linspace
    f = mylinspace(red_fc, 1.5 * (red_f_max - red_fc) + red_fc, 100)
    d = schwarz_red(f, red_fc, 1)

    # Use this endpoint in DMT case or if there is a problem with the slope finding
    s_end_pos = 0
    s_end_force = -2

    if not_DMT:
        # Find slope == red_k between vertical and horizontal parts of unstable branch
        f0 = mylinspace((7 * red_fc + 8) / 3, red_fc, 100, endpoint=False)
        d0 = schwarz_red(f0, red_fc, 0)
        # TODO: analytical gradient of schwarz
        df0dd0 = np.gradient(f0, d0)

        try:
            s_end_pos = root_scalar(lambda x: np.interp(x, d0, df0dd0) - red_k, bracket=(d0[0], d0[-1])).root
            s_end_force = np.interp(s_end_pos, d0, f0)
        except ValueError as e:
            # The named error happens when the bracket is quite small i.e. red_fc ~2
            if str(e) != 'f(a) and f(b) must have different signs':
                raise
        else:
            # This fires if there is NOT a ValueError i.e. found a good end point
            f = np.concatenate((f0, f))
            d = np.concatenate((d0, d))

    s_f = np.interp(red_delta, d, f, left=np.inf, right=0)
    s_f = np.atleast_1d(s_f).squeeze()  # don't choke on weird array shapes or scalars

    # overwrite portion after end position with tangent line
    s_f[red_delta <= s_end_pos] = (red_delta[red_delta <= s_end_pos] - s_end_pos) * red_k + s_end_force
    # alternative: solve for lj_force(x) = (x-s_end_pos)*red_k+s_end_force (rightmost soln)

    #     lj_f[red_delta>=s_end_pos] = np.inf # XXX: Why did you put this here?

    return np.minimum(s_f, lj_f)


def force_curve(red_curve, delta, k, radius, K, fc, tau,
                delta_shift, force_shift, lj_delta_scale, ):
    """Calculate a force curve from indentation data."""
    # Catch crazy inputs early
    assert k > 0, k
    assert radius > 0, radius
    assert K > 0, K
    assert fc < 0, fc
    assert 0 <= tau <= 1, tau
    assert lj_delta_scale > 0, lj_delta_scale

    # Calculate reduced/dimensionless values of inputs
    #     ref_force = gamma*np.pi*radius
    #     red_fc = fc/ref_force
    red_fc = (tau - 4) / 2  # tau = tau1**2 in Schwarz => ratio of short range to total surface energy
    ref_force = fc / red_fc
    ref_radius = (ref_force * radius / K) ** (1 / 3)
    ref_delta = ref_radius * ref_radius / radius
    red_delta = (delta - delta_shift) / ref_delta
    red_k = k / (ref_force) * (ref_delta)
    lj_delta_scale = lj_delta_scale / ref_delta

    # Match sign conventions of force curve calculations now rather than later
    red_force = red_curve(red_delta, red_fc, -red_k, -lj_delta_scale, )

    # Rescale to dimensioned units
    return (red_force * ref_force) + force_shift


def delta_curve(red_curve, force, k, radius, K, fc, tau,
                delta_shift, force_shift, lj_delta_scale, ):
    """Convenience function for inverse of force_curve, i.e. force->delta"""
    # Catch crazy inputs early
    assert k > 0, k
    assert radius > 0, radius
    assert K > 0, K
    assert fc < 0, fc
    assert 0 <= tau <= 1, tau
    assert lj_delta_scale > 0, lj_delta_scale

    # Calculate reduced/dimensionless values of inputs
    #     ref_force = gamma*np.pi*radius
    #     red_fc = fc/ref_force
    red_fc = (tau - 4) / 2  # tau = tau1**2 in Schwarz => ratio of short range to total surface energy
    ref_force = fc / red_fc
    red_force = (force - force_shift) / ref_force
    ref_radius = (ref_force * radius / K) ** (1 / 3)
    ref_delta = ref_radius * ref_radius / radius
    red_k = k / (ref_force) * (ref_delta)
    lj_delta_scale = lj_delta_scale / ref_delta

    # Match sign conventions of force curve calculations now rather than later
    red_delta = red_curve(red_force, red_fc, -red_k, -lj_delta_scale, )

    # Rescale to dimensioned units
    return (red_delta * ref_delta) + delta_shift


def schwarz_wrap(red_force, red_fc, red_k, lj_delta_scale, ):
    """So that schwarz can be directly jammed into delta_curve"""
    a = schwarz_red(red_force, red_fc, 1)
    if np.isnan(a):
        raise ValueError('nans abound', red_force, red_fc)
    return a


@np.errstate(divide='ignore', invalid='ignore')
def fitfun(z, d, k, radius, tau, sl=None):
    # Transform data to model units
    f = d * k
    delta = z - d

    # select retract or extend data to fit
    if sl is None:
        sl = slice(len(delta) // 2, None)  # retract
    #         sl = slice(len(delta)//2)      # extend
    delta = delta[sl]
    f = f[sl]

    # Very course estimate of force curve parameters for initial guess
    imin = np.argmin(f)  # TODO: better way to choose this for low adhesion
    fmin = f[imin]
    deltamin = delta[imin]
    fzero = np.median(f)
    fc_guess = fmin - fzero
    imax = np.argmax(delta)
    deltamax = delta[imax]
    fmax = f[imax]
    K_guess = (fmax - fmin) / np.sqrt(radius * (deltamax - deltamin) ** 3)
    if not np.isfinite(K_guess):
        K_guess = 1
    p0 = [K_guess, fc_guess, deltamin, fzero, 1, ]

    def partial_force_curve(delta, K, fc, delta_shift, force_shift, lj_delta_scale, ):
        return force_curve(red_retract, delta, k, radius, K, fc, tau,
                           delta_shift, force_shift, lj_delta_scale, )

    try:
        beta, cov = curve_fit(partial_force_curve, delta, f, p0=p0,
                              bounds=np.transpose((
                                  (0, np.inf),  # K
                                  (-np.inf, 0),  # fc
                                  # (0, 1),           # tau
                                  (np.min(delta), np.max(delta)),  # delta_shift
                                  (np.min(f), np.max(f)),  # force_shift
                                  (0, 100),  # lj_delta_scale
                              )),
                              xtol=1e-9, ftol=1e-8,
                              method='trf', verbose=0, jac='2-point')
    except Exception as e:
        print(str(e))
        print(p0)
        beta = np.full_like(p0, np.nan)
        cov = np.diag(beta)

    return np.concatenate((beta, np.sqrt(np.diag(cov))))


def calc_def_ind_ztru(d, beta, radius, k, tau):
    """Calculate deflection, indentation, z_true_surface given deflection data and parameters.
    
    """
    K, fc, delta_shift, force_shift, lj_delta_scale, *_ = beta

    maxforce = d[:len(d) // 25].mean() * k
    #    maxforce = d[:3].mean()*k

    maxdelta = delta_curve(schwarz_wrap, maxforce, k, radius, K, fc, tau,
                           delta_shift, force_shift, lj_delta_scale, )
    mindelta = delta_curve(schwarz_wrap, fc + force_shift, k, radius, K, fc, tau,
                           delta_shift, force_shift, lj_delta_scale, )
    zeroindforce = float(force_curve(red_retract, delta_shift, k, radius, K, fc, tau,
                                     delta_shift, force_shift, lj_delta_scale, ))

    deflection = (maxforce - (fc + force_shift)) / k
    indentation = maxdelta - mindelta
    z_true_surface = delta_shift + zeroindforce / k
    return deflection, indentation, z_true_surface
