"""
cleaning_rules.py
=================
Defines filtering rules, subject-specific exclusions, and stream selection criteria
for the CogPilot Multimodal Piloting Dataset as per project specifications.
"""

# Step 1: Streams dropped entirely
EXCLUDED_STREAMS = [
    'lslshimmerresp',
    'lslrespitrace',
    'lslshimmertorsoacc'  # Step 2: Torso-acc treated as optional / omitted
]

# Step 3: Run-level exclusions for bad ECG / EDA runs (< 90% quality)
# Format: (subject, run_folder_name)
BAD_RUNS_EXCLUSIONS = {
    # Bad ECG runs (sub-cp009)
    ('sub-cp009', 'level-01B_run-001'),
    ('sub-cp009', 'level-02B_run-003'),
    ('sub-cp009', 'level-03B_run-002'),
    ('sub-cp009', 'level-04B_run-004'),
    
    # Bad ECG runs (sub-cp026)
    ('sub-cp026', 'level-01B_run-001'),
    ('sub-cp026', 'level-02B_run-008'),
    ('sub-cp026', 'level-03B_run-002'),
    ('sub-cp026', 'level-03B_run-005'),
    ('sub-cp026', 'level-04B_run-009'),
    
    # Bad ECG run (sub-cp027)
    ('sub-cp027', 'level-02B_run-010'),
    
    # Bad EDA run (sub-cp028)
    ('sub-cp028', 'level-01B_run-012'),
}

# Step 4: Subject excluded from eye-tracking analysis ONLY (0% eye quality across all runs)
EYE_EXCLUDED_SUBJECTS = {
    'sub-cp003',
}

# Step 5: Specific bad eye-tracking runs excluded from eye analysis
EYE_EXCLUDED_RUNS = {
    ('sub-cp027', 'level-01B_run-001'),
    ('sub-cp027', 'level-03B_run-002'),
}

def is_run_excluded(subject: str, run_name: str) -> bool:
    """Checks if a run should be dropped from the main dataset."""
    return (subject, run_name) in BAD_RUNS_EXCLUSIONS

def is_eye_excluded(subject: str, run_name: str) -> bool:
    """Checks if eye-tracking features should be masked/NaN for this run."""
    if subject in EYE_EXCLUDED_SUBJECTS:
        return True
    if (subject, run_name) in EYE_EXCLUDED_RUNS:
        return True
    return False
