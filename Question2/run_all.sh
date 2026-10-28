#!/bin/sh
# Every Question 2 experiment in order, one training run at a time (12 threads).
# Safe to rerun after an interruption: finished steps are skipped, training resumes.
# Models B-D start only once predictions/Q2_11.md is committed (handout, Submission notes).
cd "$(dirname "$0")"
PY=../.venv/bin/python
LOG=logs/pipeline.log
mkdir -p logs

say() { echo "$(date '+%F %T') $*" >> $LOG; }
run() { say "run $*"; $PY -m "$@" >> logs/pipeline.out 2>&1 || { say "FAILED $*"; exit 1; }; }
train() {
    [ -f checkpoints/$1_s$2.pt ] && return 0
    say "train $1 s$2"
    $PY -m q2.train --model $1 --seed $2 --threads 12 --resume >> logs/$1_s$2.out 2>> logs/$1_s$2.err \
        || { say "FAILED train $1 s$2"; exit 1; }
}

# Wait for a training run started outside this script.
while pgrep -f "^\.\./\.venv/bin/python -m q2.train" > /dev/null; do sleep 60; done
say "start"

# Sections 2-3: model A, Table 2, Figure 2, Tables 3-4, Figures 3-5.
for s in 0 1; do train A $s; done
if [ ! -f logs/SECTION3_DONE ]; then
    run q2.run_eval A Blank CopyLast --timing
    run q2.baseline
    run q2.test_baseline
    touch logs/SECTION3_DONE
    say "section 3 done; waiting for the Q2.11 commit"
fi

# Section 4: B, C, D and the optional D+B, after Q2.11 is committed.
until git log -1 --format=%H -- predictions/Q2_11.md | grep -q .; do sleep 60; done
for m in B C D DB; do for s in 0 1; do train $m $s; done; done
run q2.run_eval B C D DB --timing
run q2.compare table5

# Section 5: symmetrised models, Table 6, Figure 6 (final model chosen in FINAL, default D).
run q2.run_eval sym-B sym-D sym-DB --timing
run q2.compare table6 --final DB
FINAL=$(cat logs/FINAL 2>/dev/null || echo D)
run q2.final --final "$FINAL"
touch logs/ALL_DONE
say "all done (final model sym-$FINAL)"
