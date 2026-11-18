"""
AI/ML timetable engine.

Pipeline:
CSV -> validation/cleaning -> feature engineering -> synthetic training data
-> model comparison -> ML assignment scores -> MILP constraint optimization
-> timetable validation and quality score.

The ML target is explicitly synthetic because the project has no historical
timetable labels. The optimizer, not the classifier, is responsible for
hard feasibility.
"""
from __future__ import annotations
import argparse, csv, json, math, os, re, sys, traceback
from pathlib import Path
from io import StringIO
from datetime import datetime
from types import SimpleNamespace

import numpy as np
import pandas as pd
import joblib
from scipy.optimize import milp, LinearConstraint, Bounds
from scipy.sparse import coo_matrix
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)
MODEL_PATH = MODEL_DIR / "best_model.joblib"
META_PATH = MODEL_DIR / "model_metadata.json"

DAYS_DEFAULT = ["Monday","Tuesday","Wednesday","Thursday","Friday"]
DAY_INDEX = {d:i for i,d in enumerate(["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday"])}
TIME_SLOTS = [(f"{h:02d}:00", f"{h+1:02d}:00") for h in [8,9,10,11,14,15,16,17]]
LAB_STARTS = [0,1,2,4,5,6]  # 2-hour blocks, no lunch overlap

TEACHER_REQUIRED = ["teacher_id","teacher_name"]
STUDENT_REQUIRED = ["division"]
ROOM_REQUIRED = ["room_number","room_type","capacity"]
ROOM_TYPE_CAPACITY_DEFAULTS = {
    "room": 100,
    "lecture": 100,
    "classroom": 100,
    "seminar": 60,
    "lab": 35,
    "laboratory": 35,
}
COMMON_COURSES = (
    ("Design and Analysis of Algorithms", "Theory", 3),
    ("Design and Analysis of Algorithms", "Lab", 2),
    ("Machine Learning", "Theory", 3),
    ("Machine Learning", "Lab", 2),
    ("Cloud Computing", "Theory", 3),
    ("Cloud Computing", "Tutorial", 1),
    ("Coursera 1", "Theory", 1),
    ("Coursera 2", "Theory", 1),
    ("DT", "Tutorial", 1),
)
CHOICE_COURSES = {
    "CN": ("Computer Networks", (("Theory", 3), ("Lab", 2))),
    "MLOPS": ("MLOPS", (("Theory", 3), ("Lab", 2))),
}
ONLINE_COURSES = {"COURSERA1", "COURSERA2", "DT"}

FEATURES = [
    "teacher_availability","teacher_workload_ratio","student_count",
    "classroom_capacity","capacity_difference","room_availability",
    "subject_frequency","subject_type","day_index","time_slot_index",
    "division_size","daily_student_hours","weekly_student_hours",
    "lab_requirement","equipment_match","teacher_preference_match"
]
CATEGORICAL = ["subject_type"]
NUMERIC = [x for x in FEATURES if x not in CATEGORICAL]

def clean_col(c):
    return re.sub(r"[^a-z0-9_]+","_",str(c).strip().lower()).strip("_")

def norm(s):
    return re.sub(r"\s+"," ",str(s or "").strip())

def parse_bool(x):
    return str(x).strip().lower() in {"1","true","yes","y","on"}

def split_pipe(x):
    return [norm(v) for v in str(x or "").split("|") if norm(v)]

def time_to_min(x):
    m = re.match(r"^(\d{1,2}):(\d{2})$", str(x).strip())
    if not m: return None
    return int(m.group(1))*60+int(m.group(2))

def faculty_time_to_min(value, is_end=False):
    raw = str(value or "").strip().lower()
    if not raw or raw in {"off", "none", "nan", "na", "n/a"}:
        return None
    match = re.match(r"^(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?$", raw)
    if not match:
        return time_to_min(raw)
    hour = int(match.group(1))
    minute = int(match.group(2) or 0)
    suffix = match.group(3)
    if suffix == "pm" and hour < 12:
        hour += 12
    elif suffix == "am" and hour == 12:
        hour = 0
    elif is_end and 1 <= hour <= 6:
        # These VIT exports use 4 / 5 / 6 for 4pm / 5pm / 6pm.
        hour += 12
    if hour > 23 or minute > 59:
        return None
    return hour * 60 + minute

def parse_ranges(value):
    out=[]
    for part in re.split(r"[|;]", str(value or "")):
        part=part.strip()
        if not part: continue
        m=re.match(r"(\d{1,2}:\d{2})\s*-\s*(\d{1,2}:\d{2})",part)
        if m: out.append((time_to_min(m.group(1)),time_to_min(m.group(2))))
    return out

def range_covers(value, start, end):
    ranges=parse_ranges(value)
    return any(a is not None and b is not None and a<=start and b>=end for a,b in ranges)

def parse_subjects(cell):
    # "DSA:3;DBMS:3" or "DSA,DBMS" accepted.
    parts=re.split(r"[;|]", str(cell or ""))
    out=[]
    for part in parts:
        part=part.strip()
        if not part: continue
        m=re.match(r"(.+?)\s*:\s*(\d+(?:\.\d+)?)$",part)
        if m:
            name=norm(m.group(1)); hours=float(m.group(2))
        else:
            name=norm(part); hours=1.0
        explicit_lab=bool(re.search(r"\[\s*(lab|practical)\s*\]",name,re.I))
        explicit_tut=bool(re.search(r"\[\s*(tutorial|tut)\s*\]",name,re.I))
        stype="Lab" if explicit_lab or re.search(r"\b(lab|practical)\b",name,re.I) else ("Tutorial" if explicit_tut else "Theory")
        name=re.sub(r"\s*\[\s*(lab|practical|tutorial|tut)\s*\]\s*","",name,flags=re.I).strip()
        if stype=="Lab":
            name=re.sub(r"\s+(lab|practical)$","",name,flags=re.I).strip()
        out.append({"name":name,"hours":hours,"type":stype})
    return out

def canonical_subject_code(value):
    return re.sub(r"[^A-Z0-9]", "", norm(value).upper())

def registered_choice(value):
    code = canonical_subject_code(value)
    if code in {"MLOPS", "MLOOPS", "MLOOP"} or "MACHINELEARNINGOPERATIONS" in code:
        return "MLOPS"
    if code in {"CN", "CNT", "COMPUTERTECHNOLOGY"} or "COMPUTERNETWORK" in code:
        return "CN"
    return None

def read_clean_csv(text, required, kind):
    if not str(text or "").strip():
        raise ValueError(f"{kind} CSV is empty.")

    # ---------------------------------------------------------
    # SPECIAL HANDLING FOR THE UPLOADED COLLEGE CSV FORMATS
    # ---------------------------------------------------------

    if kind == "Student":
        # Student exports can concatenate one table per division. Each table
        # repeats a title and header, so parse the blocks together and discard
        # those repeated rows instead of treating them as student records.
        raw_lines = str(text).splitlines()
        parsed_rows = list(csv.reader(raw_lines))
        header_index = next((i for i, row in enumerate(parsed_rows[:10])
                             if row and "div" in ",".join(row).lower()
                             and ("roll no" in ",".join(row).lower() or "prn" in ",".join(row).lower())), None)
        if header_index is None:
            df = pd.read_csv(StringIO(text))
        else:
            raw_header = parsed_rows[header_index]
            header = []
            seen_headers = {}
            for index, value in enumerate(raw_header):
                name = value.strip() or f"Unnamed: {index}"
                duplicate_index = seen_headers.get(name, 0)
                seen_headers[name] = duplicate_index + 1
                header.append(name if duplicate_index == 0 else f"{name}.{duplicate_index}")
            records = []
            section = None
            for row in parsed_rows[:header_index]:
                first = row[0].strip() if row else ""
                title = re.match(r"^(TY|AIML)\s*-?\s*([A-Z])\s*$", first, re.I)
                if title:
                    section = (title.group(1).upper(), title.group(2).upper())
            for row in parsed_rows[header_index + 1:]:
                cells = row + [""] * max(0, len(header) - len(row))
                first = cells[0].strip() if cells else ""
                title = re.match(r"^(TY|AIML)\s*-?\s*([A-Z])\s*$", first, re.I)
                if title and not any(cell.strip() for cell in cells[1:]):
                    section = (title.group(1).upper(), title.group(2).upper())
                    continue
                if not any(cell.strip() for cell in cells):
                    continue
                if first.lower() in {"div", "division"}:
                    continue
                if section and first.upper() == section[1]:
                    cells[0] = f"{section[0]}-{section[1]}"
                elif re.fullmatch(r"AIML\s*-?\s*([A-Z])", first, re.I):
                    match = re.fullmatch(r"AIML\s*-?\s*([A-Z])", first, re.I)
                    cells[0] = f"AIML-{match.group(1).upper()}"
                records.append(cells[:len(header)])
            df = pd.DataFrame(records, columns=header)

    elif kind == "Teacher":
        # Faculty CSV can contain a multi-line "Time" header.
        # Read the CSV using the first row and normalize the
        # college-specific column names below.
        raw_lines = str(text).splitlines()

        # Find the row containing the actual faculty headers.
        header_index = None
        for i, line in enumerate(raw_lines[:10]):
            lower = line.lower()
            if "name of the faculty" in lower:
                header_index = i
                break

        if header_index is not None:
            cleaned_text = "\n".join(raw_lines[header_index:])
            df = pd.read_csv(
                StringIO(cleaned_text),
                header=0,
                on_bad_lines="skip"
            )
        else:
            df = pd.read_csv(
                StringIO(text),
                header=0,
                on_bad_lines="skip"
            )

    else:
        # Classroom CSV is already a normal CSV.
        df = pd.read_csv(StringIO(text))

    # ---------------------------------------------------------
    # NORMALIZE COLUMN NAMES
    # ---------------------------------------------------------

    df.columns = [clean_col(c) for c in df.columns]

    # Make column matching easier.
    normalized = {}
    for c in df.columns:
        key = re.sub(r"[^a-z0-9]+", "_", str(c).lower()).strip("_")
        normalized[key] = c

    def find_column(names):
        for name in names:
            key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
            if key in normalized:
                return normalized[key]
        return None

    # ---------------------------------------------------------
    # FACULTY / TEACHER CSV
    # ---------------------------------------------------------

    if kind == "Teacher":

        teacher_id_col = find_column([
            "teacher_id",
            "id",
            "faculty_id",
            "sr_no",
            "sr_no_"
        ])

        teacher_name_col = find_column([
            "teacher_name",
            "name",
            "faculty_name",
            "name_of_the_faculty"
        ])

        total_load_col = find_column([
            "total_load",
            "max_hours_per_week",
            "load"
        ])

        day_col = find_column([
            "day"
        ])

        in_time_col = find_column([
            "in_time",
            "in"
        ])

        out_time_col = find_column([
            "out_time",
            "out"
        ])

        if teacher_id_col:
            df["teacher_id"] = df[teacher_id_col]

        if teacher_name_col:
            df["teacher_name"] = df[teacher_name_col]

        if total_load_col:
            df["max_hours_per_week"] = df[total_load_col]
        else:
            df["max_hours_per_week"] = 18

        # Preserve faculty availability information.
        if day_col:
            df["available_days"] = df[day_col]
        else:
            df["available_days"] = ""

        if in_time_col:
            df["available_start"] = df[in_time_col]
        else:
            df["available_start"] = ""

        if out_time_col:
            df["available_end"] = df[out_time_col]
        else:
            df["available_end"] = ""

        # Cleaned institutional exports often store one availability window
        # per weekday instead of a single Day / In / Out set.
        for day in ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]:
            day_key = day.lower()
            start_col = find_column([f"{day_key}_in", f"{day_key}_in_time", f"{day_key}_start"])
            end_col = find_column([f"{day_key}_out", f"{day_key}_out_time", f"{day_key}_end"])
            if start_col:
                df[f"availability_{day_key}_start"] = df[start_col]
            if end_col:
                df[f"availability_{day_key}_end"] = df[end_col]

    # ---------------------------------------------------------
    # STUDENT CSV
    # ---------------------------------------------------------

    elif kind == "Student":

        division_col = find_column([
            "division",
            "div",
            "class",
            "section"
        ])

        roll_col = find_column([
            "roll_no",
            "roll_number",
            "roll"
        ])

        name_col = find_column([
            "name",
            "student_name"
        ])

        subject_col = find_column([
            "subject_registred",
            "subject_registered",
            "subjects",
            "subject"
        ])
        if subject_col is None:
            subject_col = next(
                (c for c in df.columns if re.match(r"^subject_(registered|registred)_", c)),
                None
            )

        if division_col:
            df["division"] = df[division_col]

        if roll_col:
            df["roll_number"] = df[roll_col]

        if name_col:
            df["student_name"] = df[name_col]

        if subject_col:
            df["subjects"] = df[subject_col]

        if "division" in df.columns:
            df["division"] = (df["division"].astype("string").str.strip()
                              .str.replace(r"\s+", " ", regex=True)
                              .str.replace(r"^TY\s*[_-]?\s*([A-Z])$", r"TY-\1", regex=True, case=False))

        if "academic_hours_required" not in df.columns:
            df["academic_hours_required"] = 0

    # ---------------------------------------------------------
    # CLASSROOM CSV
    # ---------------------------------------------------------

    elif kind == "Classroom":

        room_col = find_column([
            "room_number",
            "room",
            "room_id"
        ])

        room_type_col = find_column([
            "room_type",
            "type"
        ])

        capacity_col = find_column([
            "capacity",
            "strength",
            "count"
        ])

        if room_col:
            df["room_number"] = df[room_col]

        if room_type_col:
            df["room_type"] = df[room_type_col]

        if capacity_col:
            df["capacity"] = df[capacity_col]
        else:
            # Capacity is absent from some cleaned room exports. Use a
            # conservative VIT default by room type and mark inferred values
            # in the cleaned report so the assumption is visible to admins.
            room_types = df["room_type"].astype(str).str.strip().str.lower()
            df["capacity"] = room_types.map(ROOM_TYPE_CAPACITY_DEFAULTS).fillna(0)
            df["capacity_inferred"] = df["capacity"] > 0

    # ---------------------------------------------------------
    # GENERIC ALIASES
    # ---------------------------------------------------------

    aliases = {
        "teacher_id": ["id", "faculty_id", "sr_no"],
        "teacher_name": ["name", "faculty_name"],
        "room_number": ["room", "room_id"],
        "room_type": ["type"],
        "student_count": ["strength", "count", "students"],
        "division": ["class", "section", "div"],
    }

    for target, alts in aliases.items():
        if target not in df.columns:
            for a in alts:
                if a in df.columns:
                    df[target] = df[a]
                    break

    # ---------------------------------------------------------
    # REQUIRED COLUMNS
    # ---------------------------------------------------------

    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"{kind} CSV missing required columns: {', '.join(missing)}"
        )

    # ---------------------------------------------------------
    # CLEANING
    # ---------------------------------------------------------

    before = len(df)

    missing_cells = int(df.isna().sum().sum())

    dup = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    if kind == "Student" and "division" in df.columns:
        df["division"] = (df["division"].astype("string").str.strip()
                          .str.replace(r"\s+", " ", regex=True)
                          .str.replace(r"^TY\s*[_-]?\s*([A-Z])$", r"TY-\1", regex=True, case=False))

    numeric_cols = [
        c for c in [
            "capacity",
            "max_hours_per_week",
            "student_count",
            "academic_hours_required",
            "roll_number"
        ]
        if c in df.columns
    ]

    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

        if df[c].isna().any():
            if not df[c].dropna().empty:
                df[c] = df[c].fillna(df[c].median())
            else:
                df[c] = df[c].fillna(0)

    for c in df.columns:
        if df[c].dtype == "object":
            df[c] = df[c].fillna("")

    # ---------------------------------------------------------
    # REMOVE INVALID RECORDS
    # ---------------------------------------------------------

    invalid = 0

    if kind == "Classroom":
        bad = (
            (df["capacity"] <= 0)
            |
            (df["room_number"].astype(str).str.strip() == "")
        )

        invalid = int(bad.sum())
        df = df.loc[~bad].copy()

    elif kind == "Teacher":
        bad = (
            (df["teacher_name"].astype(str).str.strip() == "")
            |
            (df["teacher_id"].astype(str).str.strip() == "")
        )

        invalid = int(bad.sum())
        df = df.loc[~bad].copy()

    elif kind == "Student":
        bad = df["division"].astype(str).str.strip() == ""

        invalid = int(bad.sum())
        df = df.loc[~bad].copy()

    # ---------------------------------------------------------
    # OUTLIER DETECTION
    # ---------------------------------------------------------

    outliers = 0

    for c in numeric_cols:
        s = df[c].dropna()

        if len(s) >= 4:
            q1, q3 = s.quantile([0.25, 0.75])
            iqr = q3 - q1

            if iqr > 0:
                outliers += int(
                    (
                        (s < q1 - 1.5 * iqr)
                        |
                        (s > q3 + 1.5 * iqr)
                    ).sum()
                )

    # ---------------------------------------------------------
    # REPORT
    # ---------------------------------------------------------

    report = {
        "kind": kind,
        "rows_before": before,
        "rows_after": len(df),
        "missing_values": missing_cells,
        "duplicates": dup,
        "invalid_records": invalid,
        "outliers_detected": outliers,
        "columns_processed": list(df.columns),
        "cleaned_csv": df.to_csv(index=False),
    }

    return df, report

def parse_config(config):
    c=config or {}
    days=c.get("working_days") or DAYS_DEFAULT
    days=[d for d in days if d in DAY_INDEX and d!="Sunday"]
    if not days: days=DAYS_DEFAULT[:]
    hours=int(c.get("hours_per_day") or 5)
    hours=max(1,min(10,hours))
    return {"working_days":days,"hours_per_day":hours,
            "division_settings":c.get("division_settings") or {}}

def division_records(students,rooms=None):
    records=[]
    for div,g in students.groupby("division",dropna=False):
        div=norm(div)
        count=0
        if "student_count" in g.columns:
            count=max(0,int(pd.to_numeric(g["student_count"],errors="coerce").fillna(0).max()))
        count=max(count,len(g),1)
        student_choices=[]
        for _,student in g.iterrows():
            student_choices.append({
                choice
                for subject in parse_subjects(student.get("subjects",""))
                if (choice:=registered_choice(subject["name"]))
            })

        batch_count=2 if div.upper().split("-")[-1]=="G" else 3
        batch_count=min(batch_count,count)
        batch_base,remainder=divmod(count,batch_count)
        batch_sizes=[
            batch_base+(1 if batch_index<remainder else 0)
            for batch_index in range(batch_count)
        ]
        batch_choices=[{"CN":0,"MLOPS":0} for _ in batch_sizes]
        batch_unassigned=[0 for _ in batch_sizes]
        batch_student_counts=[0 for _ in batch_sizes]
        batch_index=0
        for choices in student_choices:
            while (
                batch_index<len(batch_sizes)
                and batch_student_counts[batch_index]>=batch_sizes[batch_index]
            ):
                batch_index+=1
            if batch_index>=len(batch_sizes):
                break
            batch_student_counts[batch_index]+=1
            if choices:
                for code in choices:
                    batch_choices[batch_index][code]+=1
            else:
                batch_unassigned[batch_index]+=1

        for index,batch_size in enumerate(batch_sizes):
            batch_unassigned[index]+=batch_size-batch_student_counts[index]

        base_cohorts=[]
        for code,(name,_) in CHOICE_COURSES.items():
            choice_batch_counts=[batch[code] for batch in batch_choices]
            students_in_choice=sum(choice_batch_counts)
            if students_in_choice:
                base_cohorts.append({
                    "id":f"{div}::{code}",
                    "label":f"Registered {name} ({students_in_choice} students)",
                    "student_count":students_in_choice,
                    "choice_code":code,
                    "batch_counts":choice_batch_counts,
                    "batch_group_ids":[None for _ in batch_sizes],
                })
        unassigned_count=sum(batch_unassigned)
        if unassigned_count:
            base_cohorts.append({
                "id":f"{div}::UNASSIGNED",
                "label":"Students without a registered choice",
                "student_count":unassigned_count,
                "choice_code":None,
                "batch_counts":batch_unassigned,
                "batch_group_ids":[None for _ in batch_sizes],
            })
        if not base_cohorts:
            base_cohorts=[{
                "id":f"{div}::ALL",
                "label":"All students",
                "student_count":count,
                "choice_code":None,
                "batch_counts":batch_sizes,
                "batch_group_ids":[None for _ in batch_sizes],
            }]

        cohorts=[]
        batches=[]
        for batch_index,batch_size in enumerate(batch_sizes):
            attendance_groups=[]
            for base in base_cohorts:
                students_in_batch=base["batch_counts"][batch_index]
                if not students_in_batch:
                    continue
                cohort_id=f"{base['id']}::B{batch_index+1}"
                choice_name=(
                    CHOICE_COURSES[base["choice_code"]][0]
                    if base["choice_code"]
                    else "Students without a registered choice"
                )
                cohort={
                    "id":cohort_id,
                    "label":f"{div} · Batch {batch_index+1} · {choice_name} ({students_in_batch} students)",
                    "student_count":students_in_batch,
                    "choice_code":base["choice_code"],
                    "base_cohort_id":base["id"],
                    "batch_index":batch_index+1,
                }
                cohorts.append(cohort)
                attendance_groups.append(cohort_id)
                base["batch_group_ids"][batch_index]=cohort_id
            batches.append({
                "id":f"{div}::B{batch_index+1}",
                "label":f"{div} · Batch {batch_index+1} ({batch_size} students)",
                "student_count":batch_size,
                "attendance_groups":attendance_groups,
            })

        all_group_ids=[cohort["id"] for cohort in cohorts]
        specs=[]
        for name,stype,hours in COMMON_COURSES:
            if stype in {"Lab","Tutorial"}:
                for batch_index,batch in enumerate(batches,1):
                    specs.append({
                        "name":name,
                        "hours":hours,
                        "type":stype,
                        "attendance_groups":batch["attendance_groups"],
                        "cohort_label":f"Batch {batch_index} ({batch['student_count']} students)",
                        "student_count":batch["student_count"],
                        "choice_code":None,
                    })
            else:
                specs.append({
                    "name":name,
                    "hours":hours,
                    "type":stype,
                    "attendance_groups":all_group_ids,
                    "cohort_label":"All students",
                    "student_count":count,
                    "choice_code":None,
                })
        for base in base_cohorts:
            code=base["choice_code"]
            if not code:
                continue
            name,requirements=CHOICE_COURSES[code]
            for stype,hours in requirements:
                if stype in {"Lab","Tutorial"}:
                    for batch_index,students_in_batch in enumerate(base["batch_counts"],1):
                        if not students_in_batch:
                            continue
                        specs.append({
                            "name":name,
                            "hours":hours,
                            "type":stype,
                            "attendance_groups":[base["batch_group_ids"][batch_index-1]],
                            "cohort_label":f"{base['label']} · Batch {batch_index} ({students_in_batch} students)",
                            "student_count":students_in_batch,
                            "choice_code":code,
                        })
                else:
                    specs.append({
                        "name":name,
                        "hours":hours,
                        "type":stype,
                        "attendance_groups":[
                            group_id
                            for group_id in base["batch_group_ids"]
                            if group_id
                        ],
                        "cohort_label":base["label"],
                        "student_count":base["student_count"],
                        "choice_code":code,
                    })
        records.append({"division":div,"student_count":count,"cohorts":cohorts,"batches":batches,"subjects":specs,
                        "academic_hours_required":float(pd.to_numeric(g.get("academic_hours_required",pd.Series([0])),errors="coerce").fillna(0).max())})
    return records

def clean_training_frame(raw):
    x=raw.copy()
    x["teacher_availability"]=x["teacher_availability"].astype(float)
    x["room_availability"]=x["room_availability"].astype(float)
    x["capacity_difference"]=x["classroom_capacity"]-x["student_count"]
    return x[FEATURES]

def build_synthetic_training(teachers, students, rooms, config, seed=42, n_per=12, max_rows=60000):
    """Build bounded synthetic training data from real uploaded resources.

    Large colleges can have hundreds of faculty and many room/division/subject
    combinations. Exhaustively multiplying every combination can create millions
    of rows and make local training impractical. We therefore sample from the
    real candidate space with a deterministic cap while preserving both feasible
    and infeasible examples.
    """
    rng=np.random.default_rng(seed)
    divs=division_records(students,rooms)
    base=[]
    for d in divs:
        for subj in d["subjects"]:
            for t in teachers.itertuples():
                for r in rooms.itertuples():
                    base.append((d,subj,t,r))
    if not base:
        raise ValueError("Unable to create synthetic training candidates from the uploaded data.")
    target=max(2000, min(max_rows, len(base)*n_per))
    # Sample base combinations with replacement so training size stays bounded.
    choices=rng.integers(0,len(base),size=target)
    rows=[]
    for idx in choices:
        d,subj,t,r=base[int(idx)]
        day=str(rng.choice(config["working_days"]))
        slot=int(rng.integers(0,len(TIME_SLOTS)))
        start=time_to_min(TIME_SLOTS[slot][0]); end=time_to_min(TIME_SLOTS[slot][1])
        teacher_av=1 if teacher_available(t,day,start,end) else 0
        room_av=1 if room_available(r,day,start,end) else 0
        stype=subj["type"]
        student_count=subj["student_count"]
        attendance_groups=subj["attendance_groups"]
        weekly_hours=max(
            (
                sum(
                    subject["hours"]
                    for subject in d["subjects"]
                    if group in subject.get("attendance_groups",attendance_groups)
                )
                for group in attendance_groups
            ),
            default=0,
        )
        capdiff=float(r.capacity-student_count)
        lab_req=1 if stype=="Lab" else 0
        equipment=1 if (not lab_req or parse_bool(getattr(r,"has_lab",0))) else 0
        pref=teacher_preference(t,day,start,end)
        workload_ratio=float(rng.uniform(0.05,.9))
        subj_freq=subj["hours"]
        feasible=int(teacher_av and room_av and capdiff>=0 and (not lab_req or "lab" in str(r.room_type).lower()) and equipment)
        quality=(2.2*teacher_av+1.7*room_av+1.8*(capdiff>=0)+1.5*equipment+1.2*pref+0.8*(1-workload_ratio)+0.4*(day in config["working_days"]))
        label=int(feasible and quality>=5.0)
        if rng.random()<.04: label=1-label
        rows.append([teacher_av,workload_ratio,student_count,r.capacity,capdiff,room_av,subj_freq,stype,DAY_INDEX.get(day,0),slot,student_count,float(weekly_hours),float(weekly_hours),lab_req,equipment,pref,label])
    cols=FEATURES+["assignment_feasible"]
    frame=pd.DataFrame(rows,columns=cols)
    # Guarantee both classes for robust train/validation/test splits.
    if frame["assignment_feasible"].nunique()<2:
        frame.loc[0,"assignment_feasible"]=0
        frame.loc[1,"assignment_feasible"]=1
    return frame

def make_pipeline(model):
    pre=ColumnTransformer([
        ("num",Pipeline([("imp",SimpleImputer(strategy="median")),("scale",StandardScaler())]),NUMERIC),
        ("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),
                         ("onehot",OneHotEncoder(handle_unknown="ignore"))]),CATEGORICAL)
    ])
    return Pipeline([("preprocess",pre),("model",model)])

def train_models(teachers,students,rooms,config):
    frame=build_synthetic_training(teachers,students,rooms,config)
    X=frame[FEATURES]; y=frame["assignment_feasible"].astype(int)
    # Deterministic 70/15/15 split.
    rng=np.random.default_rng(42)
    idx=np.arange(len(frame)); rng.shuffle(idx)
    n=len(idx); ntr=max(1,int(n*.70)); nval=max(1,int(n*.15))
    train_idx=idx[:ntr]; val_idx=idx[ntr:ntr+nval]; test_idx=idx[ntr+nval:]
    if len(np.unique(y.iloc[train_idx]))<2:
        raise ValueError("Synthetic training data contains only one class; add more varied teachers/rooms.")
    models={
        "Decision Tree":DecisionTreeClassifier(max_depth=8,min_samples_leaf=3,random_state=42,class_weight="balanced"),
        # The engine runs as a child process of the API server. Spawning a
        # worker pool from that process is unreliable on Windows (and can be
        # denied in managed environments), so keep this modest model fit in
        # the current process.
        "Random Forest":RandomForestClassifier(n_estimators=160,max_depth=10,min_samples_leaf=2,random_state=42,n_jobs=1,class_weight="balanced"),
        "Gradient Boosting":GradientBoostingClassifier(random_state=42,n_estimators=120,max_depth=3)
    }
    results=[]
    fitted={}
    for name,model in models.items():
        pipe=make_pipeline(model); pipe.fit(X.iloc[train_idx],y.iloc[train_idx])
        pv=pipe.predict(X.iloc[val_idx])
        results.append({"model":name,"accuracy":float(accuracy_score(y.iloc[val_idx],pv)),
                        "precision":float(precision_score(y.iloc[val_idx],pv,zero_division=0)),
                        "recall":float(recall_score(y.iloc[val_idx],pv,zero_division=0)),
                        "f1":float(f1_score(y.iloc[val_idx],pv,zero_division=0))})
        fitted[name]=pipe
    best_name=max(results,key=lambda r:(r["accuracy"],r["f1"]))["model"]
    best=fitted[best_name]
    pt=best.predict(X.iloc[test_idx])
    prob=best.predict_proba(X.iloc[test_idx])[:,1] if hasattr(best,"predict_proba") else pt
    cm=confusion_matrix(y.iloc[test_idx],pt).tolist()
    test_metrics={"accuracy":float(accuracy_score(y.iloc[test_idx],pt)),
                  "precision":float(precision_score(y.iloc[test_idx],pt,zero_division=0)),
                  "recall":float(recall_score(y.iloc[test_idx],pt,zero_division=0)),
                  "f1":float(f1_score(y.iloc[test_idx],pt,zero_division=0)),
                  "roc_auc":float(roc_auc_score(y.iloc[test_idx],prob)) if len(np.unique(y.iloc[test_idx]))>1 else None,
                  "confusion_matrix":cm}
    joblib.dump(best,MODEL_PATH)
    # Actual feature importance after preprocessing.
    model_obj=best.named_steps["model"]; pre=best.named_steps["preprocess"]
    names=list(pre.get_feature_names_out())
    importances=getattr(model_obj,"feature_importances_",np.zeros(len(names)))
    fi=sorted([{"feature":n.replace("num__","").replace("cat__",""),"importance":float(v)} for n,v in zip(names,importances)],
              key=lambda z:z["importance"],reverse=True)[:20]
    meta={"model_version":"1.0","trained_at":datetime.now().isoformat(timespec="seconds"),
          "dataset_label":"Synthetic Training Data","rows":int(len(frame)),
          "train_rows":int(len(train_idx)),"validation_rows":int(len(val_idx)),"test_rows":int(len(test_idx)),
          "models":results,"best_model":best_name,"test_metrics":test_metrics,
          "feature_importance":fi,"features":FEATURES}
    META_PATH.write_text(json.dumps(meta,indent=2),encoding="utf-8")
    return meta

def teacher_available(t,day,start,end):
    unavailable_days=set(split_pipe(getattr(t,"unavailable_days","")))
    if day in unavailable_days: return False
    us=split_pipe(getattr(t,"unavailable_slots",""))
    if any(range_covers(v,start,end) for v in us): return False
    day_key = day.lower()
    start_field = f"availability_{day_key}_start"
    end_field = f"availability_{day_key}_end"
    if hasattr(t, start_field) or hasattr(t, end_field):
        daily_start = getattr(t, start_field, "")
        daily_end = getattr(t, end_field, "")
        available_start = faculty_time_to_min(daily_start)
        available_end = faculty_time_to_min(daily_end, is_end=True)
        if available_start is None or available_end is None:
            return False
        if start < available_start or end > available_end:
            return False
    allowed=split_pipe(getattr(t,"preferred_days",""))
    slots=split_pipe(getattr(t,"preferred_time_slots",""))
    # Preferred fields are soft; explicit unavailable fields are hard.
    return True

def teacher_preference(t,day,start,end):
    score=0.0
    days=set(split_pipe(getattr(t,"preferred_days","")))
    if days: score += .5 if day in days else 0
    slots=split_pipe(getattr(t,"preferred_time_slots",""))
    if slots and any(range_covers(v,start,end) for v in slots): score += .5
    return min(1.0,score)

def room_available(r,day,start,end):
    days=set(split_pipe(getattr(r,"available_days","")))
    if days and day not in days: return False
    slots=split_pipe(getattr(r,"available_slots",""))
    return not slots or any(range_covers(v,start,end) for v in slots)

def room_ok(r,stype,student_count):
    if int(getattr(r,"capacity",0)) < student_count: return False
    is_lab_room = "lab" in str(getattr(r,"room_type","")).lower()
    if (stype=="Lab") != is_lab_room: return False
    return True

def ensure_model(teachers,students,rooms,config):
    if not MODEL_PATH.exists() or not META_PATH.exists():
        return train_models(teachers,students,rooms,config)
    try: return json.loads(META_PATH.read_text(encoding="utf-8"))
    except Exception: return train_models(teachers,students,rooms,config)

def candidate_frame(cands):
    return pd.DataFrame(cands)[FEATURES]

def rotate_teacher_candidates(teachers, limit, rotation):
    preferences = sorted({preference for _, preference in teachers}, reverse=True)
    balanced = []
    for preference in preferences:
        group = [candidate for candidate in teachers if candidate[1] == preference]
        offset = rotation % len(group)
        balanced.extend(group[offset:] + group[:offset])
    return balanced[:limit]

def select_balanced_candidates(alternatives, limit, teacher_rotation=0, room_rotation=None):
    remaining = list(alternatives)
    if limit <= 0 or not remaining:
        return []
    if room_rotation is None:
        room_rotation = teacher_rotation

    teacher_ids = list(dict.fromkeys(c["teacher_id"] for c in remaining))
    teacher_rank = {
        teacher_id: (index - teacher_rotation) % len(teacher_ids)
        for index, teacher_id in enumerate(teacher_ids)
    }
    room_ids = list(dict.fromkeys(c["room_number"] for c in remaining))
    room_rank = {
        room_id: (index - room_rotation) % len(room_ids)
        for index, room_id in enumerate(room_ids)
    }
    selected = []
    used_teachers = set()
    used_rooms = set()

    while remaining and len(selected) < limit:
        best = max(
            remaining,
            key=lambda candidate: (
                int(candidate["teacher_id"] not in used_teachers)
                + int(candidate["room_number"] not in used_rooms),
                candidate["teacher_id"] not in used_teachers,
                candidate["room_number"] not in used_rooms,
                -teacher_rank[candidate["teacher_id"]],
                -room_rank[candidate["room_number"]],
                candidate.get("ml_score", 0),
                candidate["features"]["teacher_preference_match"],
                -candidate["features"]["capacity_difference"],
            ),
        )
        remaining.remove(best)
        selected.append(best)
        used_teachers.add(best["teacher_id"])
        used_rooms.add(best["room_number"])

    return selected

def candidate_generation(teachers, students, rooms, config, model):
    divs = division_records(students,rooms)
    tasks = []

    for d in divs:
        for subj in d["subjects"]:
            hours = float(subj["hours"])
            stype = subj["type"]
            duration = 2 if stype == "Lab" else 1
            count = max(1, int(math.ceil(hours / duration)))
            attendance_groups=subj.get("attendance_groups") or [f"{d['division']}::ALL"]
            weekly_hours=max(
                (
                    sum(
                        subject["hours"]
                        for subject in d["subjects"]
                        if group in subject.get("attendance_groups",attendance_groups)
                    )
                    for group in attendance_groups
                ),
                default=0,
            )

            for k in range(count):
                tasks.append({
                    "task_id": f"{d['division']}|{','.join(attendance_groups)}|{subj['name']}|{stype}|{k+1}",
                    "division": d["division"],
                    "student_count": subj.get("student_count",d["student_count"]),
                    "subject": subj["name"],
                    "subject_type": stype,
                    "duration": duration,
                    "attendance_groups": attendance_groups,
                    "cohort_label": subj.get("cohort_label","All students"),
                    "choice_code": subj.get("choice_code"),
                    "online": canonical_subject_code(subj["name"]) in ONLINE_COURSES,
                    "subject_frequency": hours,
                    "weekly_student_hours": weekly_hours,
                })

    if not tasks:
        raise ValueError("No timetable tasks could be created from the student data.")

    # ---------------------------------------------------------
    # PRE-FILTER RESOURCES
    # ---------------------------------------------------------
    teacher_rows = list(teachers.itertuples())
    room_rows = list(rooms.itertuples())

    if not teacher_rows:
        raise ValueError("No valid teachers are available.")

    if not room_rows:
        raise ValueError("No valid classrooms are available.")

    cands = []

    # Limit candidate combinations BEFORE ML prediction.
    # This is the main performance improvement.
    MAX_TEACHERS_PER_TASK = 12
    MAX_ROOMS_PER_TASK = 8
    MAX_CANDIDATES_PER_TIME = 1

    room_task_order = {}
    room_task_counts = {}
    for task in tasks:
        if task["online"]:
            continue
        room_type = "lab" if task["subject_type"] == "Lab" else "lecture"
        room_task_order[task["task_id"]] = room_task_counts.get(room_type, 0)
        room_task_counts[room_type] = room_task_order[task["task_id"]] + 1

    for task_index, task in enumerate(tasks):

        division_cfg = config["division_settings"].get(
            task["division"], {}
        )

        division_days = int(
            division_cfg.get(
                "days_per_week",
                len(config["working_days"])
            )
            or len(config["working_days"])
        )

        allowed_days = config["working_days"][
            :max(
                1,
                min(
                    len(config["working_days"]),
                    division_days
                )
            )
        ]

        # -----------------------------------------------------
        # PRE-CALCULATE VALID ROOMS
        # -----------------------------------------------------
        if task["online"]:
            valid_rooms=[
                SimpleNamespace(
                    room_number="Online",
                    capacity=task["student_count"],
                    room_type="online",
                    has_lab=0,
                    available_days="",
                    available_slots="",
                )
            ]
        else:
            valid_rooms = []
            for r in room_rows:
                if not room_ok(r,task["subject_type"],task["student_count"]):
                    continue
                valid_rooms.append(r)

        if not valid_rooms:
            raise ValueError(
                f"No suitable classroom found for "
                f"{task['division']} - {task['subject']} "
                f"(students={task['student_count']}, "
                f"type={task['subject_type']})."
            )

        # Prefer rooms with the smallest capacity difference.
        valid_rooms.sort(
            key=lambda r: (
                int(getattr(r, "capacity", 0))
                - task["student_count"]
            )
        )

        valid_rooms = valid_rooms[:MAX_ROOMS_PER_TASK]

        # -----------------------------------------------------
        # GENERATE CANDIDATES
        # -----------------------------------------------------
        task_candidates = []

        for day in allowed_days:

            starts = (
                LAB_STARTS
                if task["duration"] == 2
                else list(range(len(TIME_SLOTS)))
            )

            for si in starts:

                start = time_to_min(TIME_SLOTS[si][0])
                end = start + task["duration"] * 60

                if end > time_to_min("18:00"):
                    continue

                # -------------------------------------------------
                # Find teachers valid for this exact day/time.
                # -------------------------------------------------
                valid_teachers = []

                for t in teacher_rows:

                    if not teacher_available(
                        t,
                        day,
                        start,
                        end
                    ):
                        continue

                    pref = teacher_preference(
                        t,
                        day,
                        start,
                        end
                    )

                    valid_teachers.append(
                        (t, pref)
                    )

                if not valid_teachers:
                    continue

                # Preserve preference tiers while distributing the teacher
                # shortlist across tasks and time slots.
                teacher_rotation = (
                    task_index
                    + DAY_INDEX.get(day, 0) * len(TIME_SLOTS)
                    + si
                )
                valid_teachers = rotate_teacher_candidates(
                    sorted(valid_teachers, key=lambda x: x[1], reverse=True),
                    MAX_TEACHERS_PER_TASK,
                    teacher_rotation,
                )

                # -------------------------------------------------
                # Combine only the reduced teacher/room sets.
                # -------------------------------------------------
                for t, pref in valid_teachers:

                    subj_freq = task["subject_frequency"]

                    max_hours = float(
                        getattr(
                            t,
                            "max_hours_per_week",
                            18
                        )
                        or 18
                    )

                    workload_ratio = min(
                        1.0,
                        float(subj_freq) / max_hours
                    )

                    for r in valid_rooms:

                        if not room_available(
                            r,
                            day,
                            start,
                            end
                        ):
                            continue

                        equipment = (
                            1
                            if task["subject_type"] != "Lab"
                            or parse_bool(
                                getattr(r, "has_lab", 0)
                            )
                            or str(
                                getattr(r, "room_type", "")
                            ).lower() == "lab"
                            else 0
                        )

                        row = {
                            "teacher_availability": 1,
                            "teacher_workload_ratio": workload_ratio,
                            "student_count": task["student_count"],
                            "classroom_capacity": int(r.capacity),
                            "capacity_difference":
                                int(r.capacity)
                                - task["student_count"],
                            "room_availability": 1,
                            "subject_frequency":
                                float(subj_freq),
                            "subject_type":
                                task["subject_type"],
                            "day_index":
                                DAY_INDEX.get(day, 0),
                            "time_slot_index": si,
                            "division_size":
                                task["student_count"],
                            "daily_student_hours":
                                config["hours_per_day"],
                            "weekly_student_hours":task["weekly_student_hours"],
                            "lab_requirement":
                                1
                                if task["subject_type"] == "Lab"
                                else 0,
                            "equipment_match": equipment,
                            "teacher_preference_match": pref
                        }

                        task_candidates.append({
                            **task,
                            "day": day,
                            "start_index": si,
                            "time": (
                                f"{TIME_SLOTS[si][0]}–"
                                f"{TIME_SLOTS[
                                    si + task["duration"] - 1
                                ][1]}"
                            ),
                            "teacher_id":
                                str(t.teacher_id),
                            "teacher_name":
                                str(t.teacher_name),
                            "teacher_max_hours":
                                float(
                                    getattr(
                                        t,
                                        "max_hours_per_week",
                                        18
                                    )
                                    or 18
                                ),
                            "room_number":
                                str(r.room_number),
                            "features": row
                        })

        # ---------------------------------------------------------
        # No candidate = immediate useful error.
        # ---------------------------------------------------------
        if not task_candidates:
            raise ValueError(
                f"No feasible candidate assignments found for "
                f"{task['division']} - {task['subject']} "
                f"(type={task['subject_type']}, "
                f"students={task['student_count']})."
            )

        # ---------------------------------------------------------
        # Keep only a reasonable number BEFORE ML prediction.
        # ---------------------------------------------------------
        task_candidates.sort(
            key=lambda x: (
                x["day"],
                x["start_index"],
                x["room_number"]
            )
        )

        # Keep alternatives across the full week and each time slot. Taking
        # the first N chronological candidates can leave a task with options
        # on only one day, making an otherwise schedulable timetable
        # infeasible.
        by_time = {}
        for candidate in task_candidates:
            by_time.setdefault((candidate["day"], candidate["start_index"]), []).append(candidate)
        balanced_candidates = []
        for alternatives in by_time.values():
            first = alternatives[0]
            rotation = (
                task_index
                + DAY_INDEX.get(first["day"], 0) * (len(TIME_SLOTS)-1)
                + first["start_index"]
            )
            room_rotation = (
                room_task_order.get(task["task_id"], task_index)
                + DAY_INDEX.get(first["day"], 0) * (len(TIME_SLOTS)-1)
                + first["start_index"]
            )
            selected = select_balanced_candidates(
                alternatives,
                MAX_CANDIDATES_PER_TIME,
                rotation,
                room_rotation,
            )
            balanced_candidates.extend(selected)

        task_candidates = balanced_candidates

        cands.extend(task_candidates)

    # -------------------------------------------------------------
    # ML SCORING
    # -------------------------------------------------------------
    X = pd.DataFrame(
        [c["features"] for c in cands]
    )[FEATURES]

    probs = model.predict_proba(X)[:, 1]

    for c, p in zip(cands, probs):
        c["ml_score"] = float(p)

    # -------------------------------------------------------------
    # COMPACT FINAL CANDIDATE SET
    # -------------------------------------------------------------
    grouped = {}

    for c in cands:
        grouped.setdefault(
            c["task_id"],
            []
        ).append(c)

    compact = []
    task_order = {task["task_id"]: index for index, task in enumerate(tasks)}

    for task_id, arr in grouped.items():
        by_time = {}
        for candidate in arr:
            by_time.setdefault((candidate["day"], candidate["start_index"]), []).append(candidate)
        for alternatives in by_time.values():
            first = alternatives[0]
            rotation = (
                task_order[task_id]
                + DAY_INDEX.get(first["day"], 0) * (len(TIME_SLOTS)-1)
                + first["start_index"]
            )
            room_rotation = (
                room_task_order.get(task_id, task_order[task_id])
                + DAY_INDEX.get(first["day"], 0) * (len(TIME_SLOTS)-1)
                + first["start_index"]
            )
            selected = select_balanced_candidates(
                alternatives,
                MAX_CANDIDATES_PER_TIME,
                rotation,
                room_rotation,
            )
            compact.extend(selected)

    if not compact:
        raise ValueError(
            "No candidate assignments remained after ML filtering."
        )

    return tasks, compact
def teacher_preference_by_candidate(c):
    return float(c["features"]["teacher_preference_match"])

def build_milp(tasks,cands,config):
    if not cands: raise ValueError("No candidate assignments.")
    idx={i:c for i,c in enumerate(cands)}
    rows=[]; cols=[]; data=[]; lb=[]; ub=[]; rowno=0
    def add_group(items, lo, hi):
        nonlocal rowno
        for i in items: rows.append(rowno); cols.append(i); data.append(1.0)
        lb.append(lo); ub.append(hi); rowno+=1
    by_task={}
    for i,c in idx.items(): by_task.setdefault(c["task_id"],[]).append(i)
    for t in tasks: add_group(by_task.get(t["task_id"],[]),1,1)
    # resource conflicts by hour
    resources={}
    for i,c in idx.items():
        slots=range(c["start_index"],c["start_index"]+c["duration"])
        for si in slots:
            key=("teacher",c["teacher_id"],c["day"],si); resources.setdefault(key,[]).append(i)
            if not c.get("online"):
                key=("room",c["room_number"],c["day"],si); resources.setdefault(key,[]).append(i)
            for group in c.get("attendance_groups",[f"{c['division']}::ALL"]):
                key=("division",group,c["day"],si); resources.setdefault(key,[]).append(i)
    for items in resources.values(): add_group(items,0,1)
    # Daily hours are limited per attending student cohort, not per division.
    divday={}
    for i,c in idx.items():
        for group in c.get("attendance_groups",[f"{c['division']}::ALL"]):
            divday.setdefault((group,c["day"]),[]).append((i,c["duration"],c["division"]))
    for key,items in divday.items():
        for i,dur,_ in items: rows.append(rowno); cols.append(i); data.append(float(dur))
        div=items[0][2]
        limit=int(config["division_settings"].get(div,{}).get("hours_per_day",config["hours_per_day"]))
        lb.append(0); ub.append(limit); rowno+=1
    # teacher weekly hours
    tw={}
    for i,c in idx.items(): tw.setdefault(c["teacher_id"],[]).append((i,c["duration"]))
    for tid,items in tw.items():
        maxh=max(float(idx[i].get("teacher_max_hours",18) or 18) for i,_ in items)
        for i,dur in items: rows.append(rowno); cols.append(i); data.append(float(dur))
        lb.append(0); ub.append(maxh); rowno+=1
    A=coo_matrix((data,(rows,cols)),shape=(rowno,len(cands))).tocsr()
    # Maximize ML suitability plus preference, compactness proxy, and capacity fit.
    objective=[]
    for c in cands:
        capfit=min(1.0,max(0.0,c["features"]["capacity_difference"]/max(1,c["features"]["student_count"])))
        score=100*c["ml_score"]+8*c["features"]["teacher_preference_match"]+4*capfit-0.8*c["start_index"]
        objective.append(-score)

    lower=np.array(lb)
    upper=np.array(ub)
    constraints=LinearConstraint(A,lower,upper)

    def has_feasible_solution(result):
        if result.x is None:
            return False
        values=np.asarray(result.x)
        if not np.all(np.abs(values-np.round(values))<=1e-5):
            return False
        activity=A@values
        return bool(np.all(activity>=lower-1e-6) and np.all(activity<=upper+1e-6))

    result=milp(c=np.array(objective),integrality=np.ones(len(cands)),bounds=Bounds(0,1),
                constraints=constraints,options={"time_limit":5})
    if not has_feasible_solution(result):
        result=milp(c=np.zeros(len(cands)),integrality=np.ones(len(cands)),bounds=Bounds(0,1),
                    constraints=constraints,options={"time_limit":10})
    if not has_feasible_solution(result):
        raise ValueError(f"Constraint optimization did not find a feasible solution: {result.message}")
    chosen=[cands[i] for i,v in enumerate(result.x) if v>.5]
    return chosen

def validate_schedule(chosen,config,divs):
    conflicts=[]
    used=set()
    for c in chosen:
        for si in range(c["start_index"],c["start_index"]+c["duration"]):
            resources=[("Teacher",c["teacher_id"])]
            if not c.get("online"):
                resources.append(("Room",c["room_number"]))
            resources.extend(
                ("Division",group)
                for group in c.get("attendance_groups",[f"{c['division']}::ALL"])
            )
            for kind,key in resources:
                k=(kind,key,c["day"],si)
                if k in used:
                    conflicts.append({"severity":"high","category":"Conflict","title":f"{kind} conflict",
                                      "detail":f"{key} is double-booked on {c['day']} at slot {si}."})
                used.add(k)
    # Verify required hours separately for each student cohort.
    for d in divs:
        cohorts=d.get("cohorts") or [{"id":f"{d['division']}::ALL","label":d["division"]}]
        for cohort in cohorts:
            group=cohort["id"]
            expected=sum(
                subject["hours"]
                for subject in d["subjects"]
                if group in subject.get("attendance_groups",[f"{d['division']}::ALL"])
            )
            actual=sum(
                c["duration"]
                for c in chosen
                if c["division"]==d["division"]
                and group in c.get("attendance_groups",[f"{d['division']}::ALL"])
            )
            if actual+1e-6<expected:
                conflicts.append({"severity":"high","category":"Weekly hours",
                                  "title":"Weekly hours not satisfied",
                                  "detail":f"{cohort['label']} in {d['division']} requires {expected:g} hours but received {actual:g}."})
    return conflicts

def quality_score(chosen,config,conflicts):
    if not chosen: return 0.0
    hard=max(0,len(conflicts))
    ml=np.mean([c["ml_score"] for c in chosen])*100
    pref=np.mean([c["features"]["teacher_preference_match"] for c in chosen])*100
    capacity=np.mean([min(1,max(0,c["features"]["capacity_difference"]/max(1,c["student_count"]))) for c in chosen])*100
    score=0.60*ml+0.20*pref+0.20*capacity
    return float(max(0,min(100,score if hard==0 else score*max(0,1-hard/len(chosen)))))

def generate(teachers,students,rooms,config):
    meta=ensure_model(teachers,students,rooms,config)
    model=joblib.load(MODEL_PATH)
    tasks,cands=candidate_generation(teachers,students,rooms,config,model)
    chosen=build_milp(tasks,cands,config)
    divs=division_records(students,rooms)
    conflicts=validate_schedule(chosen,config,divs)
    sessions=[]
    for c in sorted(chosen,key=lambda x:(DAY_INDEX.get(x["day"],99),x["start_index"],x["division"])):
        sessions.append({"day":c["day"],"time":c["time"],"subject":c["subject"],
                         "faculty":c["teacher_name"],"room":c["room_number"],
                         "division":c["division"],"status":"draft",
                         "classType":c["subject_type"],"batch":None,
                         "batchSize":c["student_count"],"cohort":c["cohort_label"],
                         "studentCount":c["student_count"],"mlScore":round(c["ml_score"]*100,2)})
    return {"sessions":sessions,"conflicts":conflicts,"quality_score":quality_score(chosen,config,conflicts),
            "model":meta,"candidate_count":len(cands),"task_count":len(tasks)}

def inspect(teachers,students,rooms,config):
    reports=[]
    # read_clean_csv already generated reports externally.
    divs=division_records(students,rooms)
    subject_counts={}
    for d in divs:
        for s in d["subjects"]: subject_counts[s["name"]]=subject_counts.get(s["name"],0)+s["hours"]
    divisions=[]
    for d in divs:
        weekly_hours=max(
            (
                sum(
                    subject["hours"]
                    for subject in d["subjects"]
                    if cohort["id"] in subject.get("attendance_groups",[])
                )
                for cohort in d["cohorts"]
            ),
            default=0,
        )
        divisions.append({
            "division":d["division"],
            "student_count":d["student_count"],
            "subjects":len({subject["name"] for subject in d["subjects"]}),
            "weekly_hours":weekly_hours,
        })
    return {"divisions":divisions,
            "subject_frequency":subject_counts,
            "student_distribution":{d["division"]:d["student_count"] for d in divs},
            "teacher_workload":{str(r.teacher_name):float(getattr(r,"max_hours_per_week",18) or 18) for r in teachers.itertuples()},
            "room_capacity":{str(r.room_number):int(r.capacity) for r in rooms.itertuples()},
            "working_days":config["working_days"],"hours_per_day":config["hours_per_day"]}

def parse_payload(payload):
    config=parse_config(payload.get("config"))
    teachers,trep=read_clean_csv(payload.get("teachers_csv",""),TEACHER_REQUIRED,"Teacher")
    students,srep=read_clean_csv(payload.get("students_csv",""),STUDENT_REQUIRED,"Student")
    rooms,rrep=read_clean_csv(payload.get("classrooms_csv",""),ROOM_REQUIRED,"Classroom")
    # Normalize key optional columns.
    for c in ["max_hours_per_week"]:
        if c not in teachers.columns: teachers[c]=18
    for c in ["unavailable_days","unavailable_slots","preferred_days","preferred_time_slots"]:
        if c not in teachers.columns: teachers[c]=""
    for c in ["student_count","subjects","academic_hours_required"]:
        if c not in students.columns:
            students[c]=0 if c!="subjects" else ""
    for c in ["available_days","available_slots","has_lab","has_projector","has_computer"]:
        if c not in rooms.columns: rooms[c]="" if c in ["available_days","available_slots"] else 0
    teachers["max_hours_per_week"]=pd.to_numeric(teachers["max_hours_per_week"],errors="coerce").fillna(18)
    rooms["capacity"]=pd.to_numeric(rooms["capacity"],errors="coerce").fillna(0).astype(int)
    students["student_count"]=pd.to_numeric(students["student_count"],errors="coerce").fillna(0).astype(int)
    return teachers,students,rooms,config,{"teachers":trep,"students":srep,"classrooms":rrep}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("command",choices=["inspect","train","generate"])
    args=ap.parse_args()
    try:
        payload=json.load(sys.stdin)
        teachers,students,rooms,config,reports=parse_payload(payload)
        if args.command=="inspect":
            out={"ok":True,"data_quality":reports,"analytics":inspect(teachers,students,rooms,config)}
        elif args.command=="train":
            meta=train_models(teachers,students,rooms,config)
            out={"ok":True,"data_quality":reports,"model":meta}
        else:
            result=generate(teachers,students,rooms,config)
            out={"ok":True,"data_quality":reports,**result}
        print(json.dumps(out,default=lambda x: x.item() if hasattr(x,"item") else str(x)))
    except Exception as e:
        print(json.dumps({"ok":False,"error":str(e),"trace":traceback.format_exc()}))
        sys.exit(1)

if __name__=="__main__":
    main()
