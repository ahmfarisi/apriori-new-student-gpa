# Apriori Association Rules: New-Student Profiles and First-Semester GPA

Analysis code for the article *"Association Patterns between New-Student Admission Profiles and First-Semester GPA Using Apriori"* (submitted to INTENSIF: Jurnal Ilmiah Penelitian dan Penerapan Teknologi Sistem Informasi).

## What the script does
1. Builds one transaction per student from categorical attributes (school major, school type, admission path, study program, entry age, birthplace and domicile groups) plus a first-semester GPA band.
2. Mines frequent itemsets with Apriori (mlxtend), `min_support = 0.02`, `max_len = 4`, and keeps rules whose consequent is a single GPA band with `lift >= 1.10`.
3. Tests every rule with chi-square (Fisher's exact test when an expected cell < 5), applies Benjamini–Hochberg FDR correction, and removes redundant rules (confidence must be >= 1.10 × the best confidence of any sub-antecedent, including the base rate).
4. Checks stability across cohorts (lift > 1 in every intake year) and compares the two faculties.
5. Runs a sensitivity grid over `min_support` × `min_lift`.

## Usage
```bash
pip install -r requirements.txt
python apriori_analisis.py [path_to_data.xlsx] [path_to_output.xlsx]
```
By default the script reads `data/Data PMB Siap Apriori.xlsx` (sheet `Data Turunan`) and writes `Hasil Analisis Apriori.xlsx` next to the script.

## Data
The student dataset is **not included** in this repository to protect student privacy. It is available from the corresponding author on reasonable request. The input sheet must contain these columns:
`masuk_analisis, angkatan, fakultas, jurusan, JALUR, prodi, kategori_usia, kategori_ip, status_sekolah, kelompok_tempat_lahir, kelompok_domisili`.

## License
MIT
