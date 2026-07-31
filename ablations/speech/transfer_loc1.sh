#!/bin/bash
# Ablations Job-1 — transfer loc1 audio setanta -> LUMI (run ON setanta, agent-forwarded).
#
#   ssh -A setanta
#   nohup bash transfer_loc1.sh > transfer_loc1.log 2>&1 &
#
# For each loc1 source subdir listed in wavs.dur.list, rsync its top-level *.wav
# (+ wavs.dur) straight to LUMI scratch, staged into the flat <dest_label>/ layout
# that audio-tknz.py expects. The forwarded id_ed25519 authenticates the LUMI leg,
# so bytes go setanta -> LUMI directly (never through the laptop).
#
# dest_label = dirname(wavs.dur relpath) with '/' -> '_'  (matches the 10 h subset:
#   bbc/audio -> bbc_audio, soundcloud/rnl/audio -> soundcloud_rnl_audio,
#   turassiar/wavs -> turassiar_wavs, podcasts_jul24/x -> podcasts_jul24_x).
set -uo pipefail

UNLAB=/media/storage/phonetics/asr_data_irish/unlabelled
LIST="$UNLAB/wavs.dur.list"
LUMI=mcinerne@lumi.csc.fi
DEST=/scratch/project_465002364/audio/unlabelled_full

fail=0
# loc1 entries are the './'-prefixed relative ones; absolute paths are loc2 (skip).
grep '^\./' "$LIST" | while IFS= read -r rel; do
    rel="${rel#./}"                       # bbc/audio/wavs.dur
    durdir="$(dirname "$rel")"            # bbc/audio
    label="${durdir//\//_}"              # bbc_audio
    src="$UNLAB/$durdir/"
    if [ ! -d "$src" ]; then
        echo "SKIP missing src: $src"
        continue
    fi
    echo "=== $durdir -> $label ==="
    rsync -rt --info=progress2 --partial \
        -e "ssh -i $HOME/.ssh/id_lumi_ed25519 -o IdentitiesOnly=yes -o BatchMode=yes" \
        --include='*.wav' --include='*.WAV' --include='wavs.dur' --exclude='*' \
        "$src" "$LUMI:$DEST/$label/" || { echo "RSYNC FAIL: $label"; fail=1; }
done

echo "transfer done (fail=$fail)"
