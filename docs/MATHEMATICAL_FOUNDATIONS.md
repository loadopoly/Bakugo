# Mathematical & Physical Foundations

Bakugo's metrology and financial engines are built on first-principles physics and statistical estimation.

---

## 1. Optical Refraction & Stack Displacement

When measuring a card through slab acrylic and display glass, light rays bend according to Snell's Law at each dielectric interface:

$$n_1 \sin \theta_1 = n_2 \sin \theta_2$$

### In-Plane Card Surface Displacement
The textbook lateral offset formula $d = t \cdot \frac{\sin(\theta_1 - \theta_2)}{\cos \theta_2}$ measures the perpendicular distance between rays in free space. However, card metrology requires the displacement **along the card surface plane**:

$$\delta = t \cdot (\tan \theta_1 - \tan \theta_2) = \frac{d}{\cos \theta_1}$$

Using the perpendicular formula under-reports by 9% at $25^\circ$ tilt and 23% at $40^\circ$ tilt.

### Layer Stack Composition
For a multi-layer stack of $M$ layers with thickness $t_i$ and refractive index $n_i$:

$$\delta_{\text{total}} = \sum_{i=1}^M t_i \cdot (\tan \theta_{\text{incident}} - \tan \theta_{i})$$

Air gaps ($n=1$) contribute **zero** in-plane displacement since $\tan \theta_1 - \tan \theta_{\text{air}} = 0$.

---

## 2. Inverse-Variance Sensor Fusion & $\chi^2$ Inflation

Repeated measurements across video frames or multiple observers combine via minimum-variance weighting:

$$\hat{\mu} = \frac{\sum_i w_i x_i}{\sum_i w_i}, \quad w_i = \frac{1}{\sigma_i^2}$$

### Outlier Inconsistency Protection
Standard inverse-variance fusion shrinks uncertainty as $1 / \sqrt{N}$, which leads to overconfidence if glare creates biased measurements. To maintain coverage:

$$\chi^2 = \sum_i \frac{(x_i - \hat{\mu})^2}{\sigma_i^2}$$

If $\chi^2 > \text{dof}$ (where $\text{dof} = N - 1$):

$$\hat{\sigma}_{\text{combined}} = \frac{1}{\sqrt{\sum w_i}} \cdot \max\left(1.0, \sqrt{\frac{\chi^2}{\text{dof}}}\right)$$

Inconsistent measurements **widen** the reported confidence interval rather than falsely narrowing it.

---

## 3. Almgren-Chriss Optimal Inventory Liquidation

For an inventory of $X$ units liquidated over horizon $T$ divided into $N$ intervals of length $\tau$:
* **Permanent market impact**: $g(v) = \gamma v$
* **Temporary slippage**: $h(v) = \eta v$
* **Unsold inventory volatility risk**: $\sigma$

The optimal trajectory minimising $E[\text{Cost}] + \lambda \text{Var}[\text{Cost}]$ follows:

$$x(t) = X \cdot \frac{\sinh(\kappa (T - t))}{\sinh(\kappa T)}$$

Where the urgency parameter $\kappa$ is defined by:

$$\frac{2(\cosh(\kappa \tau) - 1)}{\tau^2} = \frac{\lambda \sigma^2}{\tilde{\eta}}, \quad \tilde{\eta} = \eta - \frac{\gamma \tau}{2}$$
