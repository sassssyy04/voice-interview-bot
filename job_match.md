Job Matching – Criteria & Formulas

This document captures the exact scoring logic and supporting rules implemented in JobMatchingService for ranking jobs against a candidate profile.

1) Matching Flow (High Level)

Load jobs from data/jobs.json, normalize each record to the Job model.

Distance pre‑filter (60 km):

Determine an effective candidate pincode using candidate.pincode or _approximate_pincode_from_locality().

If both candidate and job pincodes are present, compute distance via _calculate_distance() and skip jobs with distance_km > 60.

Score each remaining job via _calculate_job_match() producing component scores, strengths/concerns, a rationale, and the overall score.

Sort by overall score (desc) and return top 3.

2) Criteria Weights (Overall Score)

The overall score is a weighted sum of six component scores in [0, 1]:

overall = 0.30*location
        + 0.15*salary
        + 0.25*shift
        + 0.15*language
        + 0.10*vehicle
        + 0.05*experience

Config surfaced via _get_matching_criteria() also returns max_distance_km and salary_tolerance_percent from settings.

3) Location (component score)

Inputs: candidate.pincode, job.pincode, candidate.locality, job.locality.

3.1 Distance computation

# Pincode-only heuristic
# 1000 pincode difference ≈ 100 km  ⇒  every 10 units ≈ 1 km
pincode_diff = abs(int(pincode1) - int(pincode2))
distance_km  = min(pincode_diff / 10, 500.0)

Fallback distances for invalid/missing pincodes default to 100.0 km.

3.2 Location score rules

When both pincodes exist:

If distance_km ≤ 25 → location = 1.0 and add strength “very close”.

Else if distance_km ≤ 5 → location = 0.8 and add strength “nearby”.

Else if distance_km ≤ settings.max_distance_km → location = 0.5 and add concern about travel time.

Else → location = 0.0 and add concern “too far”.

When pincodes missing but localities exist:

Same locality (case‑insensitive) → location = 0.7 and add strength “same locality”.

Different locality → location = 0.4 and add concern “pincode missing; locality approximation”.

Pre‑filter: Jobs with distance_km > 60 are dropped before scoring.

4) Salary (component score)

Inputs: candidate.expected_salary, job.salary_min, job.salary_max.

if expected ≤ job_min:
    salary = 1.0
elif expected ≤ job_max:
    # linear interpolation within [job_min, job_max]
    range_pos = (expected - job_min) / (job_max - job_min)
    salary = 1.0 - 0.3*range_pos
else:
    overage   = (expected - job_max) / job_max
    tolerance = settings.salary_tolerance_percent / 100
    if overage ≤ tolerance:
        salary = 0.7
    else:
        salary = max(0.0, 0.7 - (overage - tolerance))

Strengths/concerns are added based on the fit (e.g., “matches expectations”, “below expected”).

5) Shift (component score)

Inputs: candidate.preferred_shift, job.required_shifts.

if preferred_shift is set and (preferred in job.required_shifts or preferred == "flexible"):
    shift = 1.0
else if preferred_shift is set:
    shift = 0.3  # and add concern listing available shifts

If no preference is provided, the shift score remains at its default (0.0) and does not contribute positively.

6) Language (component score)

Inputs: candidate.languages, job.required_languages.

match_count = |candidate ∩ job_required|
need_count  = |job_required|
language = match_count / need_count   # if need_count > 0 else 1.0

Adds strengths when high coverage; lists missing languages as concerns when coverage is low.

7) Vehicle (component score)

Inputs: job.requires_two_wheeler, candidate.has_two_wheeler.

if job.requires_two_wheeler:
    vehicle = 1.0 if candidate.has_two_wheeler else 0.0
else:
    vehicle = 1.0

8) Experience (component score)

Inputs: candidate.total_experience_months, job.min_experience_months, job.max_experience_months.

if candidate_exp < min_req:
    gap = (min_req - candidate_exp) / max(min_req, 1)
    experience = max(0.0, 1.0 - gap)
elif max_req is None or candidate_exp ≤ max_req:
    experience = 1.0
else:
    excess = (candidate_exp - max_req) / max_req
    experience = max(0.3, 1.0 - 0.5*excess)

9) Strengths, Concerns & Rationale

During scoring, human‑readable strengths and concerns are accumulated.

_generate_rationale() composes a brief message:

Intro based on overall score threshold:

≥ 0.8: “Excellent match …”

≥ 0.6: “Good match …”

otherwise: “Partial match …”

Appends top strengths and concerns, plus contact number.

10) Data Normalization (jobs.json → Job)

_normalize_job_json() applies:

Category mapping to a fixed enum (e.g., many operational categories map to manufacturing).

Shifts: cleans and remaps (e.g., early_morning → morning); defaults to ["morning"] if empty.

Languages: filters to a known set.

Numeric types: enforces ints on salary/experience.

Boolean: requires_two_wheeler coerced to bool.

is_active: only active jobs proceed.

11) Locality → Pincode Approximation

_approximate_pincode_from_locality(locality) returns a representative 6‑digit pincode using a fixed map of well‑known localities/cities (Delhi, Mumbai, Bengaluru, Kolkata). If no match, returns None.

12) Configuration Inputs

settings.max_distance_km – used in location scoring thresholds (not in the 60 km pre‑filter).

settings.salary_tolerance_percent – tolerance applied when expected salary exceeds job_max.

13) Known Quirks & Notes (faithful to current code)

Pre‑filter vs. threshold: Pre‑filter is hardcoded to 60 km, while later location scoring uses settings.max_distance_km for a mid‑tier score band.

Pincode distance heuristic: Distance uses numeric pincode difference; it is not geodesic. Large inaccuracies are possible across regions.

Order of thresholds: The ≤25 km check precedes ≤5 km, so distances ≤5 satisfy the first branch (score 1.0) and never reach the 0.8 branch. (Outcome is reasonable—closer gets the higher score—but the ordering is noteworthy.)

Missing preferences: If a candidate has no shift or language preferences listed, those components do not add positive points by default.

14) Output

The service returns a MatchingResult with:

top_matches (up to 3 JobMatch items)

total_jobs_considered

matching_criteria_used (weights and key settings)

generated_at timestamp

Each JobMatch includes component scores, final match_score, strengths, concerns, and a textual rationale.

Appendix: Symbols

≤ means "less than or equal to".

All component scores are clamped to [0.0, 1.0] by construction.

