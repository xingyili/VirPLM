#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p tmp log joblib final_outputs
rm -f joblib/H3N2_HI_data.joblib
rm -f tmp/H3N2_NHT_HA.csv tmp/H3N2_NHT_HI.csv \
      tmp/H3N2_AHT_HA.csv tmp/H3N2_AHT_HI.csv tmp/H3N2.fasta
python make_who_dataset.py --subtype H3N2

