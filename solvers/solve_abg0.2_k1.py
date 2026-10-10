#!/usr/bin/env python3
"""9 版扰动扫描 (a_p x k)。由 solve_all_ap_k.ipynb 的 code cells 直接导出。"""
import os
import matplotlib
matplotlib.use("Agg")

# this code is used to solve 2D shallow water QG equations
# D/Dt(\nabla^2psi+beta*y-f^2F*psi)=0 in dimensionless form
# rewritten as D/Dt(\nabla^2psi-f^2F*psi)+beta*v=0
# periodic boundary conditions in x and y
# domain: 0<=x<Lx, -Ly<=y<=Ly

import numpy                as np
import matplotlib
import matplotlib.pyplot    as plt 
import matplotlib.animation as animation 

# pip install --upgrade imageio-ffmpeg -i https://pypi.tuna.tsinghua.edu.cn/simple
import imageio_ffmpeg
plt.rcParams["animation.ffmpeg_path"] = imageio_ffmpeg.get_ffmpeg_exe()

def Get_truncate(var_fft):
    var_fftT                     = var_fft.copy()
    nfreqx                       = int(np.floor(2/3*(nx/2-1))) # nx is even, 2/3 rule
    var_fftT[:,nfreqx+1:-nfreqx] = 0
    nfreqy                       = int(np.floor(2/3*(ny/2-1))) # ny is even, 2/3 rule
    var_fftT[nfreqy+1:-nfreqy,:] = 0
    return var_fftT

def Get_RHS(PV_fft, UB=0.0, dQdy=0.0):
    # UB   : background zonal jet U_bg(y), 2D array (ny,nx); 0 disables it
    # dQdy : background PV gradient dQ_bg/dy, 2D array (ny,nx); 0 disables it
    # infer other variables
    ps_fft  =  PV_fft/ps2PV
    UU_fft  = -ddy*ps_fft
    VV_fft  =  ddx*ps_fft

    # 2/3 rule truncation
    PV_fftT = Get_truncate(PV_fft)
    UU_fftT = Get_truncate(UU_fft)
    VV_fftT = Get_truncate(VV_fft)
    PV_T    = np.fft.ifft2(PV_fftT).real
    UU_T    = np.fft.ifft2(UU_fftT).real
    VV_T    = np.fft.ifft2(VV_fftT).real
    # background jet advects the perturbation: total u = U_bg(y) + u'
    UU_T    = UU_T + UB

    # calculate nonlinear terms
    UUPV     = UU_T*PV_T
    UUPV_fft = np.fft.fft2(UUPV)
    VVPV     = VV_T*PV_T
    VVPV_fft = np.fft.fft2(VVPV)
    # background PV gradient advects the perturbation:  -v' * dQ_bg/dy
    VQ_fft   = np.fft.fft2(VV_T*dQdy)

    # hyperviscosity to remove small-scale turbulence
    coe      = 1e-9
    hyper    = -coe*lap**2*PV_fft

    # calculate RHS terms
    RHS      = -(ddx*UUPV_fft+ddy*VVPV_fft+beta*VV_fft+VQ_fft) + hyper
    return RHS

# set dimensionless value of all parameters
beta = 1  # dimensionless value
f0   = 1  # dimensionless value
F    = 1  # (L/Ld)^2

# grid spacing and time stepping
Lx = 10
nx = 1000
dx = Lx/nx
xg = np.linspace(0,Lx,nx,endpoint=False)

Ly = 2
ny = 400
dy = 2*Ly/ny
yg = np.linspace(-Ly,Ly,ny,endpoint=False)

T    = 50
nt   = 10000
dt   = T/nt
tg   = np.linspace(0,T,nt+1,endpoint=True)
ngap = 10
tgap = tg[::ngap]

# constant coefficient matrix
lap    = np.tile(((2*np.pi*1j*np.fft.fftfreq(ny)/dy)**2).reshape((-1, 1)),[ 1,nx]) + \
         np.tile(((2*np.pi*1j*np.fft.fftfreq(nx)/dx)**2).reshape(( 1,-1)),[ny, 1])
ddx    = np.tile((2*np.pi*1j*np.fft.fftfreq(nx)/dx).reshape((1,-1)),[ny,1])
ddy    = np.tile((2*np.pi*1j*np.fft.fftfreq(ny)/dy).reshape((-1,1)),[1,nx])
ps2PV  = lap - f0**2*F


# ============================================================
# background field: psi_background, with U(y) = -d(psi_background)/dy
# U(y) is set by the user below; psi_background is obtained by FFT
# ============================================================

# ---- tunable parameters of the background profile ----
U0 = 1.0    # amplitude of the background zonal jet (user-set)
b  = 0.25   # fixed half-width of the two boundary layers
a  = 0.2    # tunable half-width of the central linear shear region

def U_background(y):
    """piecewise U(y):  U=  (2-y)/b        for  2-b <= y <= 2
                         1.0             for    a  <= y <  2-b
                         y/a             for   -a  <= y <    a
                         -1.0            for   b-2 <= y <  -a
                         -(2+y)/b        for   -2  <= y <  b-2
       Piecewise continuous and |U| <= 1.  (The old 1/(2-y) form had a
       log-singularity at y=+-2, giving |U|~50 and forcing dt down ~28x.)
    """
    y  = np.asarray(y, dtype=float)
    out = U0 * ( 2.0 - y ) / b                                  # 2-b <= y <= 2
    out = np.where((y >= a)     & (y < 2 - b), U0 *  1.0,        out)
    out = np.where((y >= -a)    & (y < a),     U0 *  y / a,      out)
    out = np.where((y >= b - 2) & (y < -a),    U0 * (-1.0),      out)
    out = np.where(y < b - 2,                  U0 * (-(2.0+y)/b), out)
    return out

# staggered grid in y: never sample the two singular endpoints
yg_b    = -Ly + (np.arange(ny) + 0.5)*dy
Ubg     = U_background(yg_b)
Ubg    -= Ubg.mean()            # zero-mean, so psi_background is periodic

# ---- FFT: psi_background = U_fft / (-1j*ky)  since U = -dpsi/dy ----
ky   = 2*np.pi*np.fft.fftfreq(ny)/dy
Ubg_fft     = np.fft.fft(Ubg)
psi_bg_fft  = np.zeros(ny, dtype=complex)
nzmask      = ky != 0                 # k=0 mode set to zero -> periodic psi
psi_bg_fft[nzmask] = Ubg_fft[nzmask]/(-1j*ky[nzmask])
psi_bg_fft[0]      = 0.0
psi_bg      = np.fft.ifft(psi_bg_fft).real          # psi_background(y)
dpsi_bg_dy  = np.fft.ifft(1j*ky*psi_bg_fft).real    # spectral check

print('U_background : max=%+.4f  min=%+.4f  <U>=%.2e' % (Ubg.max(), Ubg.min(), Ubg.mean()))
print('psi_backgroun: max=%+.4f  min=%+.4f  <psi>=%.2e' % (psi_bg.max(), psi_bg.min(), psi_bg.mean()))
print('check  max|U_background - (-d psi_bg/dy)| = %.3e' % np.abs(Ubg - (-dpsi_bg_dy)).max())
print('check  psi_bg periodic: psi_bg[0]=%.4f  psi_bg[-1]=%.4f' % (psi_bg[0], psi_bg[-1]))

# ---- broadcast background onto the 2D grid ----
psi_bg_2D = psi_bg.reshape((-1, 1))*np.ones((1, nx))   # psi_background(y)
U_bg_2D   = Ubg   .reshape((-1, 1))*np.ones((1, nx))   # U_background(y)

# background PV (for the 'total field' view): Q_bg = (lap - f0^2 F) psi_bg
Qbg_2D    = np.fft.ifft2(ps2PV*np.fft.fft2(psi_bg_2D)).real
# background PV gradient (advects the perturbation in Get_RHS): dQ_bg/dy = -U'' + F U
dQdy_2D   = np.fft.ifft2(ddy*ps2PV*np.fft.fft2(psi_bg_2D)).real



# initialize variable
# NB: run the 'background field' cell above first, it defines
#     psi_bg_2D and U_bg_2D (psi_background and U_background on the grid)
ps_save = np.zeros((int(nt/ngap)+1,ny,nx))
PV_save = np.zeros((int(nt/ngap)+1,ny,nx))
UU_save = np.zeros((int(nt/ngap)+1,ny,nx))
VV_save = np.zeros((int(nt/ngap)+1,ny,nx))
E_save  = np.zeros(int(nt/ngap)+1)          # perturbation kinetic energy E(t)

# initial perturbation of the streamfunction (physics space)
#   psi' = A_p * sin(2*pi*k_p*x/Lx) * exp(-(y/a_p)**2)
A_p     = 1e-4
k_p     = 1
a_p     = 0.1
psi_p   = A_p*np.sin(2*np.pi*k_p*xg/Lx).reshape((1,-1))*np.exp(-(yg/a_p)**2).reshape((-1,1))
psi_fft = np.fft.fft2(psi_p)
PV_fft  = ps2PV*psi_fft           # q' = (nabla^2 - f0^2 F) psi'

# infer all variables (perturbation fields)
ps_fft         =  PV_fft/ps2PV
UU_fft         = -ddy*ps_fft
VV_fft         =  ddx*ps_fft
# linear superposition of the background field:  psi_total = psi_background + psi'
ps_tot         = np.fft.ifft2(ps_fft).real
UU_tot         = np.fft.ifft2(UU_fft).real
VV_tot         = np.fft.ifft2(VV_fft).real
PV_save[0,:,:] = np.fft.ifft2(PV_fft).real
ps_save[0,:,:] = ps_tot
UU_save[0,:,:] = UU_tot
VV_save[0,:,:] = VV_tot
E_save[0] = 0.5*np.sum(np.abs(UU_fft)**2 + np.abs(VV_fft)**2)/float(nx*ny)**2

for i in np.arange(1,nt+1): 
    # Use fourth-order Runge-Kutta scheme for time stepping
    q1      = dt*Get_RHS(PV_fft, U_bg_2D, dQdy_2D)
    q2      = dt*Get_RHS(PV_fft + q1/2, U_bg_2D)
    q3      = dt*Get_RHS(PV_fft + q2/2, U_bg_2D)
    q4      = dt*Get_RHS(PV_fft + q3, U_bg_2D)

    # Update PV
    PV_fft  = PV_fft + (q1 + 2*q2 + 2*q3 + q4)/6
    
    # infer all variables
    ps_fft  =  PV_fft/ps2PV
    UU_fft  = -ddy*ps_fft
    VV_fft  =  ddx*ps_fft

    # save variables (with background field superposed)
    if i%ngap==0:
        print('Saving variables at step ',i,'...')
        PV_save[int(i/ngap),:,:] = np.fft.ifft2(PV_fft).real
        ps_save[int(i/ngap),:,:] = np.fft.ifft2(ps_fft).real
        UU_save[int(i/ngap),:,:] = np.fft.ifft2(UU_fft).real
        VV_save[int(i/ngap),:,:] = np.fft.ifft2(VV_fft).real
        E_save[int(i/ngap)] = 0.5*np.sum(np.abs(UU_fft)**2 + np.abs(VV_fft)**2)/float(nx*ny)**2
    
print('Job finished!')

np.savez_compressed('solve_2D_nonlinear_case1_abg0.2_ap0.1_k1.npz',PV_save=PV_save,ps_save=ps_save,UU_save=UU_save,VV_save=VV_save,E_save=E_save,   \
        xg=xg,yg=yg,tgap=tgap)


npzfile = np.load('solve_2D_nonlinear_case1_abg0.2_ap0.1_k1.npz')
PV_save = npzfile['PV_save'][:]
ps_save = npzfile['ps_save'][:]
UU_save = npzfile['UU_save'][:]
VV_save = npzfile['VV_save'][:]
E_save  = npzfile[  'E_save'][:]
xg      = npzfile[     'xg'][:]
yg      = npzfile[     'yg'][:]
tgap    = npzfile[   'tgap'][:]

print(np.amax(PV_save),np.amin(PV_save))

npzfile = np.load('solve_2D_nonlinear_case1_abg0.2_ap0.1_k1.npz')
PV_save = npzfile['PV_save'][:]
ps_save = npzfile['ps_save'][:]
UU_save = npzfile['UU_save'][:]
VV_save = npzfile['VV_save'][:]
E_save  = npzfile[  'E_save'][:]
xg      = npzfile[     'xg'][:]
yg      = npzfile[     'yg'][:]
tgap    = npzfile[   'tgap'][:]

print(np.amax(PV_save),np.amin(PV_save))

figE, axE = plt.subplots(1,1,figsize=(14,8))
axE.semilogy(tgap, E_save, 'o-')
axE.set_xlabel('t'); axE.set_ylabel(r"$\langle u'^2+v'^2\rangle/2$")
axE.set_title('perturbation kinetic energy  (solve_2D_nonlinear_case1_abg0.2_ap0.1_k1)')
figE.tight_layout(); figE.savefig('solve_2D_nonlinear_case1_abg0.2_ap0.1_k1_energy.png', dpi=150); plt.close(figE)

