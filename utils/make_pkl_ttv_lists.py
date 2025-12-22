import os
import re
import pickle
from datetime import datetime, timedelta
from collections import defaultdict

# ---------- CONFIG ----------
FULL_DATA_DIR = "/home/vatsal/Dataserver/Datasets/VIL/VIL_scaled_lr_240/full_data"
OUTPUT_DIR = os.path.dirname(FULL_DATA_DIR)  # parent, used for saving lists/files

MONTH_ABBR_TO_NUM = {"JUN": 6, "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10}
FNAME_REGEX = re.compile(r"^(\d{2})([A-Z]{3})(\d{4})_(\d{6})\.npy$")
MAX_GAP = timedelta(minutes=30)
WINDOW_SIZE = 15   # you defined 25 earlier; change here if you want a different window
# ----------------------------

# group files by (year, mon_abbr, day)
day_to_entries = defaultdict(list)  # key -> list of (datetime, base_name, full_path)

for fname in os.listdir(FULL_DATA_DIR):
    if not fname.endswith(".npy"):
        continue
    m = FNAME_REGEX.match(fname)
    if not m:
        # skip names that don't match pattern
        continue
    day_str, mon_abbr, year_str, time_str = m.groups()
    mon_abbr = mon_abbr.upper()
    if mon_abbr not in MONTH_ABBR_TO_NUM:
        continue  # only JJASO
    year = int(year_str)
    day = int(day_str)
    month_num = MONTH_ABBR_TO_NUM[mon_abbr]

    hh = int(time_str[0:2]); mm = int(time_str[2:4]); ss = int(time_str[4:6])
    dt = datetime(year, month_num, day, hh, mm, ss)
    base_name = fname[:-4]  # without .npy
    full_path = os.path.join(FULL_DATA_DIR, fname)
    key = (year, mon_abbr, day)
    day_to_entries[key].append((dt, base_name, full_path))

# prepare outputs
train_chunks = []  # list of chunks, each chunk is a list of base_names (no .npy)
val_chunks = []
test_chunks = []

# also maintain chunk filenames for writing .txt lists (per earlier request)
train_chunk_fnames = []
val_chunk_fnames = []
test_chunk_fnames = []

# per-day chunk counter for naming (so names are DDMMMYYYY_001, _002, ...)
day_chunk_counter = defaultdict(int)

# sort days for deterministic output
sorted_days = sorted(day_to_entries.keys(), key=lambda k: (k[0], MONTH_ABBR_TO_NUM[k[1]], k[2]))

for (year, mon_abbr, day) in sorted_days:
    entries = day_to_entries[(year, mon_abbr, day)]
    entries.sort(key=lambda x: x[0])  # sort by datetime
    times = [e[0] for e in entries]
    bases = [e[1] for e in entries]
    paths = [e[2] for e in entries]

    n = len(entries)
    if n < WINDOW_SIZE:
        continue

    # slide window
    for start in range(0, n - WINDOW_SIZE + 1):
        win_times = times[start:start + WINDOW_SIZE]
        win_bases = bases[start:start + WINDOW_SIZE]

        # check max-gap rule
        valid = True
        for i in range(1, WINDOW_SIZE):
            if win_times[i] - win_times[i - 1] > MAX_GAP:
                valid = False
                break
        if not valid:
            continue

        # valid chunk: store base names
        # note: these are like '29SEP2023_181453' (no .npy)
        chunk = list(win_bases)

        # increment day counter and form chunk filename (for text lists)
        date_str = f"{day:02d}{mon_abbr}{year}"
        day_key = (year, mon_abbr, day)
        day_chunk_counter[day_key] += 1
        chunk_idx = day_chunk_counter[day_key]
        chunk_fname = f"{date_str}_{chunk_idx:03d}.npy"

        # assign to split
        if year == 2018:
            test_chunks.append(chunk)
            test_chunk_fnames.append(chunk_fname)
        elif year == 2020:
            val_chunks.append(chunk)
            val_chunk_fnames.append(chunk_fname)
        else:
            train_chunks.append(chunk)
            train_chunk_fnames.append(chunk_fname)

# Save lists as pickles (lists of lists)
with open(os.path.join(OUTPUT_DIR, "train_chunks.pkl"), "wb") as f:
    pickle.dump(train_chunks, f)
with open(os.path.join(OUTPUT_DIR, "val_chunks.pkl"), "wb") as f:
    pickle.dump(val_chunks, f)
with open(os.path.join(OUTPUT_DIR, "test_chunks.pkl"), "wb") as f:
    pickle.dump(test_chunks, f)

# Also write chunk-name text files (one chunk filename per line), as you previously requested
with open(os.path.join(OUTPUT_DIR, "train.txt"), "w") as f:
    for name in sorted(train_chunk_fnames):
        f.write(name + "\n")
with open(os.path.join(OUTPUT_DIR, "val.txt"), "w") as f:
    for name in sorted(val_chunk_fnames):
        f.write(name + "\n")
with open(os.path.join(OUTPUT_DIR, "test.txt"), "w") as f:
    for name in sorted(test_chunk_fnames):
        f.write(name + "\n")

# Summary print
print("Done.")
print(f"Train chunks: {len(train_chunks)}  (pkl -> {os.path.join(OUTPUT_DIR,'train_chunks.pkl')})")
print(f"Val   chunks: {len(val_chunks)}  (pkl -> {os.path.join(OUTPUT_DIR,'val_chunks.pkl')})")
print(f"Test  chunks: {len(test_chunks)}  (pkl -> {os.path.join(OUTPUT_DIR,'test_chunks.pkl')})")
print(f"train.txt / val.txt / test.txt written to {OUTPUT_DIR}")
