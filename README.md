# Seat Allocation — Approach & Settings

This document explains the approach used to generate weekly seat plans from the provided **bays** and **sub-teams** data, the **tunable parameters**, and how the model enforces a **strict (lexicographic) priority** ordering in a **single MILP solve**. It’s written to be dropped into your GitHub repo.

---

## 1) Problem statement

We must assign every sub-team to office bays across **Mon–Fri** while respecting:

* **Bay capacities** per time window (overlapping shifts),
* **Full-team seating when present** (if a sub-team comes in on a day, all its members get a seat),
* Minimize operational “frictions”: **missed days**, **splits across bays**, **number of night bays powered**, and **cross-team borrowing**.

Because some sub-teams are larger than their team’s own capacity, we pre-split them into **micro-teams** so the problem remains feasible without silently changing headcounts.

---

## 2) Data model (inputs)

* **Bays**: `bays_df[Team, BayID, Capacity]`
* **Sub-teams**: `subteams_df[Team, Subteam, Shift, Size]`
  Shifts are strings like `"09:00-18:00"`. Overnight is supported (e.g., `"22:00-07:00"`).

Helpers used:

```python
DAYS = ["Mon","Tue","Wed","Thu","Fri"]

def parse_shift_minutes(shift): ...               # parses and normalizes overnights
def is_night(interval, i, cutoff_start=18*60, cutoff_end=22*60) -> bool:
    # Night if starts ≥ 18:00, ends > 22:00, or crosses midnight
```

---

## 3) Pre-processing: split oversized sub-teams (team-total rule)

**Why:** A sub-team may exceed the sum of capacities of its own team’s bays (e.g., `C4 = 772` seats vs team capacity `696`). To avoid infeasibility, we split such teams into **micro-teams** whose sizes are ≤ the **team’s total capacity**. This keeps headcounts intact and allows concurrent seating.

Function:

```python
micro_df, mapping_df = split_by_team_total(bays_df, subteams_df)
```

* Output `micro_df` contains rows like `SubteamMicro = "C4#1", "C4#2", ..."` with sizes that fit the team’s total capacity.
* `mapping_df` records micro → original sub-team mapping for reporting.

> Design choice: We deliberately **do not** “shrink” team sizes. We keep the original people counts and split into feasible chunks.

---

## 4) Optimization model (MILP)

### Variables (per micro-team `i`, day `d`, bay `b`)

* `y[i,d] ∈ {0,1}` — micro-team present on day `d`.
* `a[i,d,b] ∈ ℤ₊` — seats of `i` placed in bay `b` on `d`.
* `x[i,d,b] ∈ {0,1}` — whether bay `b` is used by `i` on `d`.
* `z[i,d] ∈ {0,1}` — **split flag** (1 if `i` uses >1 bay on `d`).
* `w[b,d] ∈ {0,1}` — **night bay** indicator (1 if any night micro sits in `b` on `d`).
* `miss[i] ∈ ℤ₊` — **missed-days slack** (enables soft min-days to keep the model solvable and informative).

### Key constraints

1. **Min days (soft)**:
   `Σ_d y[i,d] + miss[i] ≥ min_days_required`
   (If you want hard enforcement, set `make_min_days_soft=False` and drop `miss[i]`.)

2. **Full micro present when `y=1`**:
   `Σ_b a[i,d,b] = Size[i] * y[i,d]`

3. **Split structure** (soft one-bay-per-day):
   `Σ_b x[i,d,b] ≤ 1 + (|B_i|-1) * z[i,d]`
   (If `z=0`, at most one bay; otherwise splits allowed but penalized.)

4. **Linking**:
   `a[i,d,b] ≤ Size[i] * x[i,d,b]` and `a[i,d,b] ≤ Capacity[b] * x[i,d,b]`

5. **Capacity by timeslot** (every overlapping 30/60-min slot):
   `Σ_i a[i,d,b] ≤ Capacity[b]`

6. **Night bay activation** (for night micros):
   `w[b,d] ≥ x[i,d,b]` for all night `i`

Optional: borrowing (if enabled) allows `x[i,d,b]` where `bay_team[b] ≠ sub_team[i]` (penalized).

---

## 5) Objective: **single-stage lexicographic** via **auto weights**

We optimize a **single weighted sum** but choose weights **automatically** so it **mimics strict lexicographic priorities**:

**Priority order (highest → lowest)**

1. **MissedDays**: `Σ_i miss[i]`
2. **SplitDays**: `Σ_{i,d} z[i,d]`
3. **NightBays**: `Σ_{b,d} w[b,d]`
4. **BorrowEvents**: `Σ x[i,d,b]` on non-owner bays (only if borrowing is allowed)

Two automatic schemes:

### A. **Dominance weights** (`priority_mode="dominance"`)

Compute safe upper bounds (`M_max, S_max, W_max, R_max`), then set the smallest numbers that ensure:

```
γ  > α·S_max + β·W_max + δ·R_max
α  > β·W_max + δ·R_max
β  > δ·R_max
```

This guarantees **any** improvement in a higher-priority term dominates all lower-priority regressions combined.

Helper:

```python
W = compute_safe_dominance_weights(subs, DAYS, bay_capacity, min_days_required, allow_borrow)
Objective = γ·(Σ miss) + α·(Σ z) + β·(Σ w) + δ·(Σ borrowed x)
```

### B. **Normalized + ε ladder** (`priority_mode="normalized"`)

Normalize each term to \~\[0,1] and use tiny decreasing weights (e.g., `1.0, 1-1e-3, 1-2e-3, ...`).
This is **numerically stable** and fast, while practically enforcing the same order.

Helper:

```python
W = compute_normalized_epsilon_weights(...)
Objective = Γ·(Σ miss/M_max) + A·(Σ z/S_max) + B·(Σ w/W_max) + Δ·(Σ borrowed x/R_max)
```

---

## 6) Night clustering & energy awareness

* **Night bay variable `w[b,d]`** records whether a bay is active at night on a given day.
* Penalizing `Σ w[b,d]` with weight **β** makes the solver **pack night micro-teams into fewer bays**, reducing lighting/HVAC costs.
* If your policy allows **cross-team sharing at night**, set `allow_borrow=True` and keep **β ≫ δ** so the model prefers **one shared bay** over **many half-empty bays**.

---

## 7) Tunable parameters (what to change & why)

| Parameter                    | Where                              |                 Typical values | Effect                                                                                       |
| ---------------------------- | ---------------------------------- | -----------------------------: | -------------------------------------------------------------------------------------------- |
| `min_days_required`          | `run_team_total_pipeline_weighted` |                            2–4 | Policy on minimum in-office days per micro-team. Higher ⇒ harder; more pressure on capacity. |
| `make_min_days_soft`         | same                               |                   `True/False` | `True` keeps model feasible; you’ll see `MissedDays` instead of failures.                    |
| `slot_minutes`               | same                               |                       30 or 60 | Time resolution for capacity checks. 60 ⇒ fewer constraints ⇒ faster.                        |
| `allow_borrow`               | same                               |                   `True/False` | Allow cross-team bay usage. If `True`, borrowing is penalized (δ term).                      |
| `priority_mode`              | same                               | `"dominance"` / `"normalized"` | Choose strictness vs numeric stability. Both avoid manual weight tuning.                     |
| `cutoff_start`, `cutoff_end` | `is_night`                         |                  18:00 / 22:00 | Change what counts as “night” to match policy.                                               |

> You **don’t** need to set weights manually; the code computes them from instance size and `allow_borrow`.

---

## 8) How to run (minimal)

```python
# 1) Preprocess (split by team-total capacity)
micro_df, mapping_df = split_by_team_total(bays_df, subteams_df)

# 2) Solve with automatically computed lexicographic weights
schedule_micro_df = solve_microteams_weighted(
    bays_df, micro_df,
    min_days_required=3,
    make_min_days_soft=True,      # keep True to get a plan + diagnostics
    allow_borrow=False,           # set True if cross-team seating is allowed
    slot_minutes=30,
    priority_mode="dominance",    # or "normalized"
    verbose=True
)

display(schedule_micro_df)
```

---

## 9) Interpreting the schedule

* **Columns Mon–Fri**: either `"Remote"` or bay allocations like `Seat_C4(82) + Seat_C6(76)`.
* **MissedDays**: days the micro-team fell short of the policy minimum (due to capacity pressure).
* Fewer `+` signs per day means fewer **splits** (better).
* Fewer distinct night bay IDs used across the week means better **night clustering**.

---

## 10) Why the model “works”

* **Feasibility by design**: splitting into micro-teams ensures every chunk can fit within its team capacity; missed-day slack prevents hard infeasibility.
* **Full-team presence**: if a micro-team is in office on a day, every member gets a seat (`Σ a = Size·y`).
* **Capacity-aware across time**: overlapping shifts checked by `slot_minutes` time slices.
* **Lexicographic priorities** in **one pass**: auto-computed weights enforce strict ordering without multi-stage re-solves.
* **Energy-aware**: `w[b,d]` + β biases toward fewer “night-on” bays.

---

## 11) Extensibility

* **Night sharing policy**: encourage cross-team co-location at night by enabling borrowing and keeping β ≫ δ (or use different δ for night vs day).
* **Hard caps**: add a cap on total night bays per day, or limit borrowing per team.
* **Aggregation**: post-process `micro_df` → original sub-teams for executive summaries.

---

## 12) Key functions in this repo (as used)

* `parse_shift_minutes(shift)` — parse & normalize shift windows (supports overnight).
* `is_night(interval, i, cutoff_start, cutoff_end)` — flags night micro-teams.
* `split_by_team_total(bays_df, subteams_df)` — **the only** splitting rule used here.
* `compute_safe_dominance_weights(...)` — auto dominance weights for strict priority.
* `compute_normalized_epsilon_weights(...)` — normalized ladder (stable, fast).
* `solve_microteams_weighted(...)` — builds and solves the MILP with auto weights.

---

## 13) Notes & disclaimers

* If you set `make_min_days_soft=False` and the instance is over-constrained, the solver can be **infeasible** (by policy).
* Runtime grows with: number of micro-teams, candidate bays per micro, and time resolution. If needed, use `slot_minutes=60` and disable borrowing for speed.
* The dominance approach may create **large** coefficients; if numerics look unstable, switch to `"normalized"`.

---

