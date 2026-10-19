# Methodology note — IFRS 9 ECL engine

*A one-page note of the kind you would table at a model committee. Keep it this
short; the detail lives in the code and the validation notebook.*

## 1. Scope and default definition

Portfolio of loans. **Default** is fixed up front as charge-off / 90+ days past
due and held constant throughout. Any change to this definition invalidates
comparisons across runs.

## 2. Parameters

| Parameter | Approach | Key assumption |
| --- | --- | --- |
| PD | Logistic scorecard (official) + GBM challenger, both calibrated (isotonic) | PDs are calibrated probabilities, not ranking scores |
| LGD | `1 - recoveries/EAD` where available, else 45% unsecured (Basel foundation style) | Segment LGD documented where data is missing |
| EAD | Outstanding balance + CCF x undrawn limit | CCF = 0.5 for undrawn lines |

## 3. Staging and SICR

- **Stage 3**: in default (flag or ≥ 90 dpd).
- **Stage 2 (SICR)**: triggered if *any* of — current 12m PD ≥ 2× origination PD
  (relative), current − origination PD ≥ 3pp (absolute), or ≥ 30 dpd (backstop).
- **Stage 1**: otherwise.

The SICR relative threshold is the single most provision-sensitive choice and is
exposed as a dashboard control.

## 4. ECL computation

`ECL = PD × LGD × EAD`, using the **12-month** PD in Stage 1 and the **lifetime**
PD in Stage 2. In Stage 3, PD = 1, so ECL = LGD × EAD.

## 5. Forward-looking overlay

PDs are scaled by a macro sensitivity (illustrative elasticity to unemployment)
across three scenarios — baseline / adverse / upside — and the reported ECL is
the **probability-weighted** average. Scenario weights are an explicit expert
judgement and are the first thing to challenge.

## 6. Validation summary

Discrimination (AUC / Gini / KS), calibration (predicted vs observed by bucket),
stability (PSI train vs test), and backtesting by grade. The challenger is kept
only if its AUC uplift over the interpretable scorecard is material; otherwise
the scorecard stands, for governance reasons.

## 7. Known limitations

LGD/EAD uncertainty; a simplistic macro link; no Basel IRB bridge yet. These are
deliberately listed rather than hidden.
