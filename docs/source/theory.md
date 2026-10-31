# Theoretical background

This page summarizes the equations solved by `mie_perso`, from the Mie coefficients
of a single sphere to the scattering matrix of a polydisperse population of
particles. The notation follows Bohren and Huffman (1983, B&H hereafter), which is
also the reference of the underlying [PyMieScatt](https://pymiescatt.readthedocs.io)
package (Sumlin et al., 2018).

## Historical context

The theory was published by Gustav Mie in 1908 under the title *Beiträge zur Optik
trüber Medien, speziell kolloidaler Metallösungen* (Contributions to the optics of
turbid media, particularly colloidal metal suspensions). Its aim was to explain the
colours of colloidal gold suspensions measured by his student W. Steubing. Mie
expanded the field scattered by a homogeneous sphere into a series of electric and
magnetic "partial waves", each obtained by solving Maxwell's equations with the
boundary conditions on the particle surface. A very similar solution had been
derived earlier by L. Lorenz from his own theory of light, which is why the theory
is also called *Lorenz–Mie* theory (Mishchenko and Travis, 2008).

As emphasized by Mishchenko and Travis (2008), the Mie solution belongs to
frequency-domain macroscopic electromagnetics:

- all fields vary harmonically in time; the B&H convention $\exp(-i\omega t)$ is used
  here, so that an absorbing material has a refractive index with a **positive**
  imaginary part;
- the total field in the presence of the particle is split into an incident and a
  scattered part, $\mathbf{E}(\mathbf{r}) = \mathbf{E}^{inc}(\mathbf{r}) + \mathbf{E}^{sca}(\mathbf{r})$.
  This split is a mathematical procedure, not a physical "collision" of photons with
  the particle;
- the scattered field becomes an outgoing spherical wave at large distance from the
  particle (far-field zone), which is the regime used by all the quantities computed
  here.

Exact and fast, the Mie solution underlies most radiative transfer computations in
the atmosphere and the ocean. It reproduces the Rayleigh law for small particles,
the rainbows and glories of water droplets, and the fine "butterfly" polarization
structure of monodisperse spheres, which is smoothed out by size averaging
(Hansen and Travis, 1974; Mishchenko and Travis, 2008).

## Problem definition

A homogeneous sphere of radius $r$ (diameter $d = 2r$) is illuminated by a plane
wave of wavelength $\lambda$ **in the surrounding medium**,
$\lambda = \lambda_0 / n_{med}$, with $\lambda_0$ the wavelength in vacuum and
$n_{med}$ the (real) refractive index of the medium. The problem only depends on two
dimensionless parameters:

$$
x = k r = \frac{2 \pi r}{\lambda} = \frac{\pi d}{\lambda},
\qquad
m = \frac{N_{p}}{n_{med}} = n + i k',
$$ (eq-size-param)

where $k = 2\pi/\lambda$ is the wavenumber in the medium, $x$ is the **size parameter**
and $m$ is the complex refractive index of the particle $N_p$ **relative to the
medium**. Absorption corresponds to $k' > 0$.

```{note}
For particles in water, $\lambda$ is the wavelength in water and $m$ the refractive
index relative to water (e.g. $m \simeq 1.05$ for phytoplankton,
$\simeq 1.15$–$1.20$ for minerals).
```

## Mie coefficients

The scattered field is expanded in vector spherical harmonics whose weights are the
Mie coefficients $a_n$ (electric, or transverse magnetic, partial waves) and $b_n$
(magnetic, or transverse electric, partial waves). Using the Riccati–Bessel functions

$$
\psi_n(\rho) = \rho\, j_n(\rho), \qquad \xi_n(\rho) = \rho\, h_n^{(1)}(\rho),
$$ (eq-riccati)

with $j_n$ and $h_n^{(1)}$ the spherical Bessel and Hankel functions of the first
kind, and the logarithmic derivative $D_n(\rho) = \psi_n'(\rho)/\psi_n(\rho)$, the
Mie coefficients read (B&H Eqs. 4.88):

$$
a_n = \frac{\left[\dfrac{D_n(mx)}{m} + \dfrac{n}{x}\right]\psi_n(x) - \psi_{n-1}(x)}
           {\left[\dfrac{D_n(mx)}{m} + \dfrac{n}{x}\right]\xi_n(x) - \xi_{n-1}(x)},
\qquad
b_n = \frac{\left[m D_n(mx) + \dfrac{n}{x}\right]\psi_n(x) - \psi_{n-1}(x)}
           {\left[m D_n(mx) + \dfrac{n}{x}\right]\xi_n(x) - \xi_{n-1}(x)}.
$$ (eq-an-bn)

$D_n$ is computed by downward recurrence (B&H Eq. 4.89), which is numerically
stable for any $m$:

$$
D_{n-1}(\rho) = \frac{n}{\rho} - \frac{1}{D_n(\rho) + n/\rho}.
$$ (eq-dn)

The series are truncated after (Wiscombe, 1980)

$$
n_{max} = x + 4 x^{1/3} + 2
$$ (eq-nmax)

terms, so that the computation time increases linearly with $x$.

For small spheres ($x < 0.5$ in `Processor.ScatMat`), only the first terms of the
low-frequency expansion are kept (B&H p. 131; Sumlin et al., 2018, Eq. S4):

$$
\begin{aligned}
a_1 &= -\frac{2 i x^3}{3}\frac{m^2-1}{m^2+2}
       - \frac{2 i x^5}{5}\frac{(m^2-2)(m^2-1)}{(m^2+2)^2}
       + \frac{4 x^6}{9}\left(\frac{m^2-1}{m^2+2}\right)^2, \\
a_2 &= -\frac{i x^5}{15}\frac{m^2-1}{2m^2+3}, \qquad
b_1 = -\frac{i x^5}{45}(m^2-1), \qquad b_2 = 0.
\end{aligned}
$$ (eq-low-freq)

The resonances of $a_n$ and $b_n$ (sharp minima of their denominators as functions
of $x$ or $m$) are the morphology-dependent resonances responsible for the ripple
structure of the Mie solution (Mishchenko and Travis, 2008).

## Amplitude scattering matrix

In the far field, the scattered field is related to the incident field through the
amplitude scattering matrix (B&H Eq. 3.12 and 4.74). For a sphere, it is diagonal:

$$
\begin{pmatrix} E_{\parallel s} \\ E_{\perp s} \end{pmatrix}
= \frac{e^{ik(R - z)}}{-ikR}
\begin{pmatrix} S_2(\theta) & 0 \\ 0 & S_1(\theta) \end{pmatrix}
\begin{pmatrix} E_{\parallel i} \\ E_{\perp i} \end{pmatrix},
$$ (eq-amplitude)

where $\theta$ is the scattering angle, $R$ the distance to the particle and
$\parallel$, $\perp$ refer to the scattering plane. The amplitude functions are

$$
S_1(\theta) = \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} \left(a_n \pi_n + b_n \tau_n\right),
\qquad
S_2(\theta) = \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} \left(a_n \tau_n + b_n \pi_n\right),
$$ (eq-s1-s2)

with the angular functions $\pi_n = P_n^1(\mu)/\sin\theta$ and
$\tau_n = \mathrm{d}P_n^1(\mu)/\mathrm{d}\theta$ ($\mu = \cos\theta$, $P_n^1$ the associated
Legendre functions), computed by upward recurrence from $\pi_0 = 0$ and $\pi_1 = 1$:

$$
\pi_n = \frac{2n-1}{n-1}\,\mu\,\pi_{n-1} - \frac{n}{n-1}\,\pi_{n-2},
\qquad
\tau_n = n\,\mu\,\pi_n - (n+1)\,\pi_{n-1}.
$$ (eq-pi-tau)

The intensities scattered for incident light polarized perpendicular ($|S_1|^2$) and
parallel ($|S_2|^2$) to the scattering plane, and unpolarized ($SU$), are returned by
PyMieScatt `ScatteringFunction`, {py:meth}`mie_perso.Processor.S1S2Function` and
{py:meth}`mie_perso.Processor.SF_SD_mp` as

$$
SL(\theta) = |S_1|^2, \qquad SR(\theta) = |S_2|^2, \qquad SU(\theta) = \tfrac{1}{2}(SR + SL).
$$ (eq-intensities)

```{note}
Eq. S8 of Sumlin et al. (2018) swaps the $SL$ and $SR$ labels with respect to the
PyMieScatt code; the definitions above follow the code.
```

## Scattering (Mueller) matrix

The Stokes vectors $(I, Q, U, V)$ of the scattered and incident light are related by
the scattering matrix (B&H Eq. 4.77):

$$
\begin{pmatrix} I_s \\ Q_s \\ U_s \\ V_s \end{pmatrix}
= \frac{1}{k^2 R^2}
\begin{pmatrix}
S_{11} & S_{12} & 0 & 0 \\
S_{12} & S_{11} & 0 & 0 \\
0 & 0 & S_{33} & S_{34} \\
0 & 0 & -S_{34} & S_{33}
\end{pmatrix}
\begin{pmatrix} I_i \\ Q_i \\ U_i \\ V_i \end{pmatrix}.
$$ (eq-mueller)

Because of the spherical symmetry, only four elements are independent:

$$
\begin{aligned}
S_{11} &= \tfrac{1}{2}\left(|S_2|^2 + |S_1|^2\right), &
S_{12} &= \tfrac{1}{2}\left(|S_2|^2 - |S_1|^2\right), \\
S_{33} &= \tfrac{1}{2}\left(S_2^* S_1 + S_2 S_1^*\right), &
S_{34} &= \tfrac{i}{2}\left(S_1 S_2^* - S_2 S_1^*\right),
\end{aligned}
$$ (eq-sij)

with $S_{11}^2 = S_{12}^2 + S_{33}^2 + S_{34}^2$ for a single sphere. These
dimensionless elements are returned by {py:meth}`mie_perso.Processor.ScatMat` and
{py:meth}`mie_perso.Processor.ScatMat_mp` as `S11`, `S12`, `S33` and `S34`.
For unpolarized incident light, the degree of linear polarization of the scattered
light is $-S_{12}/S_{11}$.

## Cross sections and efficiency factors

The cross sections $C$ (area) are expressed through the efficiency factors
$Q = C / (\pi r^2)$ (B&H Eqs. 4.61–4.62; Sumlin et al., 2018, Eq. S2):

$$
\begin{aligned}
Q_{ext} &= \frac{2}{x^2} \sum_{n=1}^{n_{max}} (2n+1)\, \mathrm{Re}\{a_n + b_n\}, \\
Q_{sca} &= \frac{2}{x^2} \sum_{n=1}^{n_{max}} (2n+1) \left(|a_n|^2 + |b_n|^2\right), \\
Q_{abs} &= Q_{ext} - Q_{sca}, \\
Q_{back} &= \frac{1}{x^2} \left| \sum_{n=1}^{n_{max}} (2n+1)(-1)^n (a_n - b_n) \right|^2,
\qquad Q_{ratio} = \frac{Q_{back}}{Q_{sca}},
\end{aligned}
$$ (eq-qeff)

and the asymmetry parameter $g = \langle \cos\theta \rangle$ and the radiation
pressure efficiency are

$$
g = \frac{4}{Q_{sca} x^2} \left[ \sum_{n=1}^{n_{max}} \frac{n(n+2)}{n+1}
\mathrm{Re}\{a_n a_{n+1}^* + b_n b_{n+1}^*\}
+ \sum_{n=1}^{n_{max}} \frac{2n+1}{n(n+1)} \mathrm{Re}\{a_n b_n^*\} \right],
\qquad
Q_{pr} = Q_{ext} - g\, Q_{sca}.
$$ (eq-g)

These are returned by {py:meth}`mie_perso.Processor.Qeff_mp`. For large spheres,
$Q_{ext} \to 2$: the extinction cross section is twice the geometrical cross section
(the "extinction paradox", van de Hulst, 1957).

In the Rayleigh limit ($x \le 0.05$ in PyMieScatt `MieQ`), the efficiencies reduce to
(B&H Eqs. 5.8–5.9):

$$
Q_{sca} = \frac{8 x^4}{3} \left| \frac{m^2-1}{m^2+2} \right|^2, \qquad
Q_{abs} = 4 x\, \mathrm{Im}\left\{ \frac{m^2-1}{m^2+2} \right\}, \qquad
Q_{back} = \tfrac{3}{2} Q_{sca}, \qquad g = 0.
$$ (eq-rayleigh)

## Phase function and normalization

The differential scattering cross section of a sphere for unpolarized light is
$\mathrm{d}C_{sca}/\mathrm{d}\Omega = S_{11}/k^2$, so that

$$
C_{sca} = \int_{4\pi} \frac{S_{11}(\theta)}{k^2}\, \mathrm{d}\Omega
= \frac{\lambda^2}{2\pi} \int_0^\pi S_{11}(\theta) \sin\theta\, \mathrm{d}\theta .
$$ (eq-csca-s11)

Defining the scattering matrix in area units, $F_{ij} = (\lambda^2/\pi)\, S_{ij}$,
gives

$$
\frac{1}{2} \int_0^\pi F_{11}(\theta) \sin\theta\, \mathrm{d}\theta = C_{sca},
$$ (eq-f11-norm)

and the phase function $p(\theta) = F_{11}(\theta) / C_{sca}$ is normalized as
$\frac{1}{2}\int_0^\pi p(\theta)\sin\theta\,\mathrm{d}\theta = 1$, the convention of the
OSOAA radiative transfer code used in the tutorial. The asymmetry parameter is
then $g = \frac{1}{2}\int_0^\pi p(\theta)\cos\theta\sin\theta\,\mathrm{d}\theta$.

### Angular resolution of the forward peak

For $x \gg 1$, about half of $C_{sca}$ is carried by diffraction, concentrated in a
forward peak
$S_{11} \simeq x^4 \left[J_1(x\theta)/(x\theta)\right]^2$ (Fraunhofer diffraction by a disk of radius $r$),
with a first zero at $\theta = 3.83/x$ followed by rings of period $\pi/x$. On a
regular grid of step $\Delta\theta$, the trapezoidal rule only satisfies
{eq}`eq-f11-norm` for $x \ll 1/\Delta\theta$: with $\Delta\theta = 0.05°$, the
integral of $F_{11}$ falls to about half of $\langle C_{sca} \rangle$ for millimetre
particles in the visible, and normalizing $F_{11}$ by its own integral would then
nearly double the side and backward scattering. The forward part of the matrix is therefore
computed on a fine geometric grid ({py:func}`mie_perso.size_average.forward_grid`,
step of 1.3 % of the angle up to 2°) and merged with the regular grid
({py:func}`mie_perso.size_average.merge_forward`).

## Polydisperse particles

Natural particles are polydisperse. For a number size distribution $n(r)$
(particles per unit volume and per unit radius), the mean optical properties of a
particle of the population are obtained by averaging over $n(r)$, the size parameter
$x(r) = 2\pi r / \lambda$ being taken at each radius:

$$
\langle C \rangle = \frac{\int_{r_{min}}^{r_{max}} Q(x(r))\, \pi r^2\, n(r)\, \mathrm{d}r}
                         {\int_{r_{min}}^{r_{max}} n(r)\, \mathrm{d}r},
\qquad
F_{ij}(\theta) = \frac{\lambda^2}{\pi}
\frac{\int_{r_{min}}^{r_{max}} S_{ij}(x(r), \theta)\, n(r)\, \mathrm{d}r}
     {\int_{r_{min}}^{r_{max}} n(r)\, \mathrm{d}r},
$$ (eq-polydisperse)

for $C = C_{ext}, C_{sca}, C_{abs}$, so that {eq}`eq-f11-norm` still holds with
$\langle C_{sca} \rangle$. The bulk coefficients (per unit length) of a medium
containing $N$ particles per unit volume are $\beta = N \langle C \rangle$
(Sumlin et al., 2018, Eq. S5), and the mean asymmetry parameter is weighted by the
scattering (Eq. S6):

$$
\langle g \rangle = \frac{\int g(r)\, Q_{sca}(r)\, \pi r^2 \, n(r)\, \mathrm{d}r}
                         {\int Q_{sca}(r)\, \pi r^2 \, n(r)\, \mathrm{d}r}.
$$ (eq-g-mean)

Since $S_{ij}$ and $Q$ only depend on $x$ for a given $m$, one look-up table computed
over $x$ with {py:meth}`mie_perso.Processor.ScatMat_mp` and
{py:meth}`mie_perso.Processor.Qeff_mp` covers all the wavelengths and radii: the
radii are recovered by $r = x \lambda / (2\pi)$ (see the tutorial).

## Size distributions

The particle size distributions available in {py:class}`mie_perso.PSD` are, with $r$ the
radius:

- **Normal**:
  $n(r) = \dfrac{1}{\sigma\sqrt{2\pi}} \exp\left[-\dfrac{(r-\mu)^2}{2\sigma^2}\right]$;
- **Log-normal** of number median radius $r_{n,med}$ and geometric standard deviation
  $\sigma$ (in natural log):

  $$
  n(r) = \frac{1}{r\, \sigma \sqrt{2\pi}}
  \exp\left[-\frac{(\ln r - \ln r_{n,med})^2}{2\sigma^2}\right];
  $$ (eq-lognorm)

- **Junge (power law)** between $r_{min}$ and $r_{max}$, of slope $\xi$ (e.g.
  $\xi = -4$ for oceanic particles):
  $n(r) \propto r^{\xi}$ for $r_{min} \le r \le r_{max}$, 0 elsewhere;
- **Modified power law**: $n(r) \propto (r/r_{min})^{\xi}$ for
  $r_{min} \le r \le r_{max}$, constant below $r_{min}$ and 0 above $r_{max}$;
- **Microplastics** (Aoki and Furue, 2021), with $d = 2r$:
  $n(r) \propto \dfrac{d^{-4}}{\exp\left[1/(\gamma d)\right] - 1}$.

The power-law and microplastic distributions are normalized to
$\int n(r)\,\mathrm{d}r = 1$.

For the log-normal distribution, the median and modal radii of the number, surface
and volume distributions are related by

$$
\ln r_{v,med} = \ln r_{n,med} + 3\sigma^2, \qquad
\ln r_{mod} = \ln r_{med} - \sigma^2, \qquad
r_{n,med} = r_{v,mod}\, e^{-2\sigma^2},
$$ (eq-lognorm-conv)

implemented in `PSD.rnmed2rvmed`, `PSD.rmed2rmod`, `PSD.rmod2rmed` and
`PSD.rvmod2rnmed`.

### Effective radius and variance

The size distribution is summarized by its mean radius and by the effective radius
and variance of Hansen and Travis (1974), computed by {py:class}`mie_perso.SizeParam`:

$$
r_{mean} = \frac{\int r\, n(r)\, \mathrm{d}r}{\int n(r)\, \mathrm{d}r},
\qquad
G = \int \pi r^2\, n(r)\, \mathrm{d}r,
$$ (eq-rmean)

$$
r_{eff} = \frac{1}{G} \int r\, \pi r^2\, n(r)\, \mathrm{d}r,
\qquad
v_{eff} = \frac{1}{r_{eff}^2\, G} \int (r - r_{eff})^2\, \pi r^2\, n(r)\, \mathrm{d}r,
$$ (eq-reff)

where $G$ is the geometrical cross section per particle (for $\int n\,\mathrm{d}r = 1$).
The effective radius is the area-weighted mean radius, i.e. the radius that matters
for scattering by large particles. For a log-normal distribution,
$r_{eff} = r_{n,med}\, e^{5\sigma^2/2}$ and $v_{eff} = e^{\sigma^2} - 1$.

## References

- Aoki, K., and Furue, R. (2021). A model for the size distribution of marine
  microplastics: A statistical mechanics approach. *PLoS ONE*, 16(11), e0259781.
  <https://doi.org/10.1371/journal.pone.0259781>
- Bohren, C. F., and Huffman, D. R. (1983). *Absorption and Scattering of Light by
  Small Particles*. John Wiley & Sons.
- Hansen, J. E., and Travis, L. D. (1974). Light scattering in planetary atmospheres.
  *Space Science Reviews*, 16, 527–610. <https://doi.org/10.1007/BF00168069>
- Mie, G. (1908). Beiträge zur Optik trüber Medien, speziell kolloidaler
  Metallösungen. *Annalen der Physik*, 330(3), 377–445.
  <https://doi.org/10.1002/andp.19083300302>
- Mishchenko, M. I., and Travis, L. D. (2008). Gustav Mie and the evolving discipline
  of electromagnetic scattering by particles. *Bulletin of the American
  Meteorological Society*, 89(12), 1853–1861.
  <https://doi.org/10.1175/2008BAMS2632.1>
- Sumlin, B. J., Heinson, W. R., and Chakrabarty, R. K. (2018). Retrieving the aerosol
  complex refractive index using PyMieScatt: A Mie computational package with
  visualization capabilities. *Journal of Quantitative Spectroscopy and Radiative
  Transfer*, 205, 127–134. <https://doi.org/10.1016/j.jqsrt.2017.10.012>
- van de Hulst, H. C. (1957). *Light Scattering by Small Particles*. John Wiley & Sons.
- Wiscombe, W. J. (1980). Improved Mie scattering algorithms. *Applied Optics*,
  19(9), 1505–1509. <https://doi.org/10.1364/AO.19.001505>
