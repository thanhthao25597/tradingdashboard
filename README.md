# SSI Trading Analytics Dashboard

## Auto Setup Classification

Dashboard automatically classifies each trade based on market structure at the time of entry.

The objective is to identify which trading styles generate the best results.

### Breakout

Conditions:

* Entry price >= 98% of 20-day high
* 20-day return > 3%
* Volume vs MA20 >= 1.3

Interpretation:

* Buying near a recent breakout level
* Momentum and volume confirmation present

---

### Breakout / Momentum

Conditions:

* Entry price >= 98% of 20-day high
* 20-day return > 3%
* No volume confirmation

Interpretation:

* Momentum entry
* Breakout without strong volume support

---

### Pullback

Conditions:

* Price >= MA60
* Price <= MA20 × 1.03
* 20-day return >= -3%

Interpretation:

* Buying a dip within an existing trend
* Trend remains intact

---

### Reversal

Conditions:

* Price <= 105% of 20-day low
* 5-day return > 2%

Interpretation:

* Buying a recovery from a recent bottom
* Potential trend change

---

### Counter-Trend Bounce

Conditions:

* Price below MA200
* 5-day return > 3%

Interpretation:

* Short-term rebound inside a long-term downtrend
* Higher risk setup

---

### Trend Following

Conditions:

* Price >= MA200
* MA20 > MA60
* 20-day return > 5%

Interpretation:

* Following an established uptrend

---

### Falling Knife / Early Reversal

Conditions:

* 20-day return < -10%

Interpretation:

* Buying into a sharp decline
* Early reversal attempt

---

### Range Trade / Other

Conditions:

* Does not fit any category above

Interpretation:

* Sideways or unclear setup

---

### Unclassified

Conditions:

* Insufficient market data

Interpretation:

* Not enough information to classify

---

# Auto Market Regime Classification

Dashboard automatically classifies the market environment at the time of entry.

The objective is to understand which market environments produce the best results.

---

### Bull / Uptrend

Conditions:

* Price >= MA200
* MA20 > MA60
* 20-day return >= 0

Interpretation:

* Healthy uptrend
* Positive momentum

---

### Bear / Downtrend

Conditions:

* Price < MA200
* MA20 < MA60
* 20-day return <= 0

Interpretation:

* Established downtrend
* Negative momentum

---

### Recovery

Conditions:

* MA20 > MA60
* Price < MA200

Interpretation:

* Recovering from a prior downtrend
* Uptrend not yet confirmed

---

### Distribution / Weakening

Conditions:

* MA20 < MA60
* Price > MA200

Interpretation:

* Long-term trend still intact
* Short-term trend weakening
* Potential distribution phase

Typical progression:

Bull → Weakening → Distribution → Bear

---

### Range / Neutral

Conditions:

* Does not fit any category above

Interpretation:

* Sideways market
* No strong directional bias

---

### Insufficient Data

Conditions:

* Missing MA20, MA60 or price history

Interpretation:

* Unable to determine regime

---

# Auto Entry Type

Provides additional context about the entry.

Possible tags:

* Near 20D High
* Near 20D Low
* Above MA20
* Below MA20
* Above MA60
* Below MA60
* Above MA200
* Below MA200
* Volume Surge

Example:

Near 20D High, Above MA20, Above MA60, Volume Surge

This is typically consistent with a Breakout setup.

---

# Auto Confidence

Represents data completeness rather than prediction accuracy.

### High

* 7-8 indicators available

### Medium

* 5-6 indicators available

### Low

* 3-4 indicators available

### Very Low

* Less than 3 indicators available

---

# Performance Metrics

### Trades

Number of trades in the category.

### Win Rate

Percentage of profitable trades.

### Actual P&L

Total realized profit and loss.

### Average Return

Average return per trade.

### Profit Factor

Gross Profit ÷ Gross Loss.

### Average Holding Days

Average number of holding days.

---

# Interpretation Guidelines

Avoid drawing conclusions from small samples.

Recommended minimum:

* 20+ trades per category for directional insights
* 50+ trades per category for higher confidence

The most valuable analysis is typically:

* Auto Setup Performance
* Auto Market Regime Performance
* Setup × Regime Matrix

These help answer:

> Which setup works best?

and

> Which setup works best under which market conditions?
