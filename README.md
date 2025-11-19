## Math & definitions (symbol‑perfect)

Let **C** be the storage capacity (resource limit). Let \(t_p\) be the **peak year of the rate** \(Q(t)\), and \(t_n\) the **first (left) inflection** of \(Q(t)\).

| Quantity | Logistic (paper) | Gompertz (this work) |
|---|---|---|
| Cumulative \(P(t)\) | \(\displaystyle \frac{C}{1+\exp\{r(t_p-t)\}}\) | \(\displaystyle C\,\exp\!\bigl[-\exp\{-r(t-t_p)\}\bigr]\) |
| Rate \(Q(t)\) | \(\displaystyle \frac{Cr\,e^{r(t_p-t)}}{(1+e^{r(t_p-t)})^2}\) | \(\displaystyle Cr\,e^{-r(t-t_p)}\,e^{-e^{-r(t-t_p)}} \;=\; r\,P\,\ln\frac{C}{P}\) |
| Peak year of \(Q\) | \(t_p\) | \(t_p\) |
| Peak rate | \(\displaystyle \frac{Cr}{4}\) | \(\displaystyle \frac{Cr}{e}\) |
| First rate‑inflection \(t_n\) | \(\displaystyle t_p-\frac{\ln(2+\sqrt{3})}{r}\) | \(\displaystyle t_p-\frac{\ln\bigl(\tfrac{3+\sqrt{5}}{2}\bigr)}{r}\) |

**Parameterizations used in code (equivalent):**  
- Logistic: \(P(t)=\dfrac{C}{1+k\,e^{-r(t-t_0)}},\ k=\dfrac{C-S_0}{S_0}\), with \(t_p=t_0+\dfrac{\ln k}{r}\).  
- Gompertz: \(P(t)=C\,e^{-b\,e^{-r(t-t_0)}},\ b=\ln\frac{C}{S_0}\), with \(t_p=t_0+\dfrac{\ln b}{r}\).  
- Inflection offsets: \(\alpha=\ln(2+\sqrt{3}),\ \beta=\ln\!\bigl(\tfrac{3+\sqrt{5}}{2}\bigr)\), so \(t_n=t_p-\alpha/r\) (Logistic) and \(t_n=t_p-\beta/r\) (Gompertz).