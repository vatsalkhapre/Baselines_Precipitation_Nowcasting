#!/bin/bash
# Transfer FACL-trained baseline models to the archive location.
#
#   usage: transfer_facl_models.sh copy      # copy only, never deletes
#          transfer_facl_models.sh verify    # checksum every file at src vs dst
#          transfer_facl_models.sh delete    # ONLY runs if verify passes first
#
# WHAT COUNTS AS "FACL-TRAINED" (this distinction matters):
#   INCLUDED - trained with FALFCL/FACL:
#     ConvLSTM, TrajGRU, PhyDNet, MAU, SimVP, EarthFormer, EarthFarseer, AlphaPre
#     exPreCast  (uses FACL natively)
#     DiffCast   (FACL on the deterministic backbone; diffusion residual native)
#   EXCLUDED:
#     WADEPre    - trained on its OWN native loss and curriculum, by protocol.
#                  It is not a FACL model and is deliberately NOT transferred.
#     DAWN-Cast  - the proposed model, not a baseline.
set -u
DST="/home/vatsal/Dataserver2/ICLR26/Unaliased_dataset/Other_models/FACL Loss"
MODE=${1:-help}
MANIFEST=/home/vatsal/NWM/Baselines_Precipitation_Nowcasting/baseline_work/transfer_manifest.txt

case $MODE in
  copy)
    [ -s "$MANIFEST" ] || { echo "no manifest at $MANIFEST"; exit 1; }
    mkdir -p "$DST"
    while read -r host src rel; do
      [ -z "${host:-}" ] && continue
      echo "COPY  $host:$src  ->  $DST/$rel"
      mkdir -p "$DST/$rel"
      if [ "$host" = "local" ]; then
        rsync -a --info=stats2 "$src/" "$DST/$rel/" | tail -2
      else
        rsync -a -e "ssh -o BatchMode=yes" --info=stats2 "vatsal@$host:$src/" "$DST/$rel/" | tail -2
      fi
    done < "$MANIFEST"
    ;;
  verify)
    fail=0
    while read -r host src rel; do
      [ -z "${host:-}" ] && continue
      if [ "$host" = "local" ]; then
        s=$(cd "$src" && find . -type f -exec md5sum {} + 2>/dev/null | sort -k2 | md5sum | cut -d' ' -f1)
        n=$(cd "$src" && find . -type f | wc -l)
      else
        s=$(ssh -o BatchMode=yes -n vatsal@$host "cd '$src' && find . -type f -exec md5sum {} + 2>/dev/null | sort -k2 | md5sum | cut -d' ' -f1")
        n=$(ssh -o BatchMode=yes -n vatsal@$host "cd '$src' && find . -type f | wc -l")
      fi
      d=$(cd "$DST/$rel" 2>/dev/null && find . -type f -exec md5sum {} + 2>/dev/null | sort -k2 | md5sum | cut -d' ' -f1)
      m=$(cd "$DST/$rel" 2>/dev/null && find . -type f | wc -l)
      if [ "$s" = "$d" ] && [ -n "$s" ]; then echo "  OK    $rel  ($n files, md5 ${s:0:12})"
      else echo "  FAIL  $rel  src=$n/${s:0:12} dst=$m/${d:0:12}"; fail=1; fi
    done < "$MANIFEST"
    exit $fail
    ;;
  delete)
    echo "Re-verifying before any deletion..."
    if ! "$0" verify; then echo "VERIFY FAILED - NOTHING DELETED"; exit 1; fi
    while read -r host src rel; do
      [ -z "${host:-}" ] && continue
      echo "DELETE $host:$src"
      if [ "$host" = "local" ]; then rm -rf "$src"
      else ssh -o BatchMode=yes -n vatsal@$host "rm -rf '$src'"; fi
    done < "$MANIFEST"
    echo "originals deleted (verify passed first)"
    ;;
  *) sed -n '2,25p' "$0" ;;
esac
