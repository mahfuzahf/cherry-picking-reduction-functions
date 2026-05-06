import glob
import csv

files = sorted(glob.glob("data/sample_*.csv"))

with open("data/final_results.csv", "w", newline='') as out:
    writer = csv.writer(out)
    writer.writerow(["sample_id", "alpha", "mt", "cost"])

    for file in files:
        with open(file) as f:
            reader = csv.reader(f)
            next(reader)  # skip header
            for row in reader:
                writer.writerow(row)