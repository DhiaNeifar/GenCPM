"""GenCPM: adversarial Cooperative Perception Messages (CPMs), attacks, and receiver-side defenses.

Packages
  cpm         GenCPM   build CPMs from per-frame detections and apply scheduled attacks
  attacks     MalCPM   eight single-attacker attacks and four coordinated attacks
  defenses    ResilienCPM  rules, temporal models, gradient-boosted trees, PILD
  data        caches of the released frames and of the generated CPMs, content labels, folds
  evaluation  end-to-end late fusion (C++ core), detection metrics, attack sweeps
  report      tables and figures of the paper
  perception  optional: LiDAR detection for new datasets (needs OpenCOOD)

Command line: python -m gencpm --help
"""

__version__ = "1.0.0"
