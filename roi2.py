# =========================
# INPUTS (edit these)
# =========================
F = 1_296_000        # Total course cost (₹)
f = 0.95             # Loan fraction (fixed 95% => 0.95)
r_annual = 0.125     # Annual interest rate (decimal), e.g., 0.125 for 12.5%
d_years = 4          # Course duration in years
S0_annual = 250_000  # Starting salary (annual, ₹)
 
# Salary rules
inc = 0.10                 # +10% increment per year
promoB = 0.15              # +15% promotion bump
promo_after_years = 2      # promotion applies after completing 2 years (i.e., from Year-3)
 
# Tenor logic
tenor_strategy = "tiered"  # "tiered" or "smooth"
N_override = None          # e.g., 84 to force 7 years; keep None to auto-pick
 
# Search horizon for breakeven
max_projection_years = 30
 
# =========================
# HELPERS
# =========================
def auto_tenor_months(loan_amt, strategy="tiered"):
    """Repayment tenor (months) from loan size."""
    if N_override is not None:
        return int(N_override)
    if strategy == "smooth":
        # ~60 months at ~₹3L; grows with ln(loan); clamp to [60, 180]
        N = 60 + 12 * math.log(max(loan_amt, 1.0) / 300_000.0)
        return int(max(60, min(180, round(N))))
    # Tiered defaults (common India ranges)
    if loan_amt <= 500_000:   return 60    # 5y
    if loan_amt <= 1_000_000: return 84    # 7y
    if loan_amt <= 2_000_000: return 120   # 10y
    if loan_amt <= 4_000_000: return 144   # 12y
    return 180                              # 15y
 
def fmt_years_months(total_months):
    years = total_months // 12
    months = total_months % 12
    return f"{years} year{'s' if years!=1 else ''} and {months} month{'s' if months!=1 else ''}"
 
def inr(n):
    return f"{n:,.0f}"
 
# =========================
# CALCULATION
# =========================
loan = f * F
d_m = d_years * 12
r_m = r_annual / 12.0
 
# Auto tenor from loan size
N = auto_tenor_months(loan, tenor_strategy)
 
# Principal at graduation (approx: mid-program accrual of interest)
P = loan * ((1 + r_m) ** (d_m / 2.0))
 
# EMI
if r_m == 0:
    EMI = P / N
else:
    EMI = P * (r_m * (1 + r_m) ** N) / ((1 + r_m) ** N - 1)
 
# Family out-of-pocket to recover
U = (1 - f) * F
 
# Simulate months after graduation until cumulative surplus >= U
cum = 0.0
y = 1
m_after_grad = 0
horizon_months = max_projection_years * 12
 
while cum < U and m_after_grad < horizon_months:
    gross_month = (S0_annual / 12.0) * ((1 + inc) ** (y - 1))
    if y > promo_after_years:       # promo from Year-3 onward
        gross_month *= (1 + promoB)
 
    surplus = gross_month - EMI     # no tax/living costs
    cum += surplus
 
    m_after_grad += 1
    if m_after_grad % 12 == 0:
        y += 1
 
# =========================
# OUTPUT
# =========================
print("Loan Amount (95% of fees): ₹", inr(round(loan)))
print("Tenor in years:", f"{N/12:.2f}")
 
print("EMI in Rs.: ₹", inr(round(EMI)))
 
if cum < U:
    print("Years after graduation to break even: No breakeven within projection horizon")
    print("Total Breakeven time since the start of the course in years: No breakeven within projection horizon")
else:
    yg = fmt_years_months(m_after_grad)
    total_months = d_m + m_after_grad
    ts = fmt_years_months(total_months)
    print("Years after graduation to break even:", yg)
    print("Total Breakeven time since the start of the course in years:", ts)




# OUTPUT
# # =========================
# Loan Amount (95% of fees): ₹ 1,231,200
# Tenor in years: 10.00
# EMI in Rs.: ₹ 23,111
# Years after graduation to break even: 3 years and 3 months
# Total Breakeven time since the start of the course in years: 7 years and 3 months