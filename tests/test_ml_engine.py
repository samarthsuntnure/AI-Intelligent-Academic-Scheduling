import json
import sys
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"ml_engine"))
import engine

SAMPLE=ROOT/"ml_engine"/"sample_data"

def load():
    payload={
      "teachers_csv":(SAMPLE/"teachers.csv").read_text(),
      "students_csv":(SAMPLE/"students.csv").read_text(),
      "classrooms_csv":(SAMPLE/"classrooms.csv").read_text(),
      "config":{"working_days":["Monday","Tuesday","Wednesday","Thursday","Friday"],"hours_per_day":5}
    }
    return engine.parse_payload(payload)

class TimetableMLTests(unittest.TestCase):
    def test_csv_validation_and_cleaning(self):
        teachers,students,rooms,config,reports=load()
        self.assertEqual(len(teachers),6)
        self.assertEqual(len(students),2)
        self.assertEqual(len(rooms),5)
        self.assertIn("cleaned_csv",reports["teachers"])

    def test_missing_column_rejected(self):
        with self.assertRaises(ValueError):
            engine.read_clean_csv("teacher_name\nA\n",engine.TEACHER_REQUIRED,"Teacher")

    def test_registered_choice_header_adds_common_courses(self):
        csv_text = "\n".join([
            "TY A",
            "Div,Roll No.,PRN,Name,Subject Registred Computer Network / MLOPS/Game Development,,,,",
            "A,1,P1,Student One,Computer Network,,,,",
            "A,2,P2,Student Two,MLOOPS,,,,",
        ])
        students, _ = engine.read_clean_csv(csv_text,engine.STUDENT_REQUIRED,"Student")
        division = engine.division_records(students)[0]

        self.assertEqual(division["division"],"TY-A")
        self.assertEqual(division["student_count"],2)
        self.assertEqual(
            {(subject["name"],subject["type"],subject["hours"])
             for subject in division["subjects"] if not subject["choice_code"]},
            set(engine.COMMON_COURSES),
        )
        self.assertEqual(
            {cohort["choice_code"]:cohort["student_count"] for cohort in division["cohorts"]},
            {"CN":1,"MLOPS":1},
        )
        tutorials=[subject for subject in division["subjects"] if subject["type"]=="Tutorial"]
        self.assertEqual(len(tutorials),len(division["batches"])*2)
        self.assertTrue(all(subject["hours"]==1 for subject in tutorials))
        cohort_batches={
            cohort["id"]:cohort["batch_index"]
            for cohort in division["cohorts"]
        }
        self.assertTrue(all(
            len({cohort_batches[group] for group in subject["attendance_groups"]})==1
            for subject in tutorials
        ))
        self.assertNotIn("Data Structures",{subject["name"] for subject in division["subjects"]})
        self.assertNotIn("Database Systems",{subject["name"] for subject in division["subjects"]})

    def test_divisions_split_into_three_balanced_batches_except_g(self):
        students=engine.pd.DataFrame({
            "division":["TY-A"]*69+["TY-G"]*48,
            "subjects":[""]*117,
        })

        divisions={d["division"]:d for d in engine.division_records(students)}

        self.assertEqual(
            [batch["student_count"] for batch in divisions["TY-A"]["batches"]],
            [23,23,23],
        )
        self.assertEqual(
            [batch["student_count"] for batch in divisions["TY-G"]["batches"]],
            [24,24],
        )
        self.assertEqual(
            len([s for s in divisions["TY-A"]["subjects"] if s["type"]=="Tutorial"]),
            6,
        )
        self.assertEqual(
            len([s for s in divisions["TY-G"]["subjects"] if s["type"]=="Tutorial"]),
            4,
        )

    def test_choice_cohorts_can_overlap_but_core_classes_cannot(self):
        divs=[{
            "division":"D1",
            "cohorts":[
                {"id":"D1::CN","label":"CN"},
                {"id":"D1::MLOPS","label":"MLOPS"},
            ],
            "subjects":[],
        }]
        choices=[
            {"teacher_id":"T1","room_number":"R1","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10},"attendance_groups":["D1::CN"]},
            {"teacher_id":"T2","room_number":"R2","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10},"attendance_groups":["D1::MLOPS"]},
        ]
        config={"working_days":["Monday"],"hours_per_day":5}
        self.assertEqual(engine.validate_schedule(choices,config,divs),[])

        core=dict(choices[0],teacher_id="T3",room_number="R3",attendance_groups=["D1::CN","D1::MLOPS"])
        conflicts=engine.validate_schedule(choices+[core],config,divs)
        self.assertTrue(any(conflict["title"]=="Division conflict" for conflict in conflicts))

    def test_model_training_and_metrics(self):
        teachers,students,rooms,config,_=load()
        meta=engine.train_models(teachers,students,rooms,config)
        self.assertIn(meta["best_model"],{"Decision Tree","Random Forest","Gradient Boosting"})
        self.assertEqual(meta["train_rows"]+meta["validation_rows"]+meta["test_rows"],meta["rows"])
        self.assertIn("f1",meta["test_metrics"])
        self.assertTrue(meta["feature_importance"])

    def test_generation_has_no_hard_conflicts(self):
        teachers,students,rooms,config,_=load()
        result=engine.generate(teachers,students,rooms,config)
        self.assertGreater(len(result["sessions"]),0)
        self.assertEqual(result["conflicts"],[])
        self.assertGreater(result["quality_score"],0)

    def test_division_specific_days(self):
        teachers,students,rooms,config,_=load()
        config["division_settings"]={"CSE-A":{"days_per_week":3,"hours_per_day":8}}
        result=engine.generate(teachers,students,rooms,config)
        days={s["day"] for s in result["sessions"] if s["division"]=="CSE-A"}
        self.assertLessEqual(len(days),3)

    def test_feature_engineering_candidate_columns(self):
        teachers,students,rooms,config,_=load()
        model=engine.joblib.load(engine.MODEL_PATH)
        tasks,cands=engine.candidate_generation(teachers,students,rooms,config,model)
        self.assertTrue(tasks)
        self.assertTrue(cands)
        self.assertTrue(all(set(engine.FEATURES).issubset(c["features"]) for c in cands[:10]))

    def test_candidate_selection_balances_resources_and_rotates_teachers(self):
        candidates=[
          {
            "teacher_id":teacher,
            "room_number":room,
            "features":{"teacher_preference_match":1,"capacity_difference":0},
          }
          for teacher in ("T1","T2","T3","T4","T5","T6")
          for room in ("R1","R2","R3","R4")
        ]

        first=engine.select_balanced_candidates(candidates,3,teacher_rotation=0)
        selected=engine.select_balanced_candidates(candidates,3,teacher_rotation=3)

        self.assertEqual({candidate["teacher_id"] for candidate in selected},{"T4","T5","T6"})
        self.assertEqual(len({candidate["room_number"] for candidate in selected}),3)
        self.assertNotEqual(
            {candidate["room_number"] for candidate in first},
            {candidate["room_number"] for candidate in selected},
        )

    def test_teacher_shortlist_rotates_within_preference_tiers(self):
        teachers=[(f"T{index}",1) for index in range(30)]

        first={teacher for teacher,_ in engine.rotate_teacher_candidates(teachers,12,0)}
        second={teacher for teacher,_ in engine.rotate_teacher_candidates(teachers,12,12)}

        self.assertEqual(len(first),12)
        self.assertEqual(len(second),12)
        self.assertNotEqual(first,second)

    def test_model_prediction_scores(self):
        teachers,students,rooms,config,_=load()
        model=engine.joblib.load(engine.MODEL_PATH)
        tasks,cands=engine.candidate_generation(teachers,students,rooms,config,model)
        self.assertTrue(all(0 <= c["ml_score"] <= 1 for c in cands[:50]))

    def test_teacher_conflict_detection(self):
        chosen=[{"teacher_id":"T1","room_number":"R1","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}},
                {"teacher_id":"T1","room_number":"R2","division":"D2","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}}]
        conflicts=engine.validate_schedule(chosen,{"working_days":["Monday"],"hours_per_day":5},[])
        self.assertTrue(any("Teacher conflict"==c["title"] for c in conflicts))

    def test_classroom_conflict_detection(self):
        chosen=[{"teacher_id":"T1","room_number":"R1","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}},
                {"teacher_id":"T2","room_number":"R1","division":"D2","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}}]
        conflicts=engine.validate_schedule(chosen,{"working_days":["Monday"],"hours_per_day":5},[])
        self.assertTrue(any("Room conflict"==c["title"] for c in conflicts))

    def test_division_conflict_detection(self):
        chosen=[{"teacher_id":"T1","room_number":"R1","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}},
                {"teacher_id":"T2","room_number":"R2","division":"D1","day":"Monday","start_index":0,"duration":1,"ml_score":.8,"features":{"teacher_preference_match":1,"capacity_difference":10,"student_count":10}}]
        conflicts=engine.validate_schedule(chosen,{"working_days":["Monday"],"hours_per_day":5},[])
        self.assertTrue(any("Division conflict"==c["title"] for c in conflicts))

    def test_weekly_hours_validation(self):
        chosen=[]
        divs=[{"division":"D1","student_count":20,"subjects":[{"name":"A","hours":3,"type":"Theory"}]}]
        conflicts=engine.validate_schedule(chosen,{"working_days":["Monday"],"hours_per_day":5},divs)
        self.assertTrue(any(c["category"]=="Weekly hours" for c in conflicts))

    def test_quality_score_is_bounded(self):
        teachers,students,rooms,config,_=load()
        result=engine.generate(teachers,students,rooms,config)
        self.assertGreaterEqual(result["quality_score"],0)
        self.assertLessEqual(result["quality_score"],100)

    def test_capacity_filtering(self):
        teachers,students,rooms,config,_=load()
        rooms=rooms.copy()
        rooms["capacity"]=10
        with self.assertRaises(ValueError):
            engine.generate(teachers,students,rooms,config)

    def test_room_type_matches_class_type(self):
        lecture=engine.SimpleNamespace(room_type="Lecture",capacity=50)
        lab=engine.SimpleNamespace(room_type="Lab",capacity=50)

        self.assertTrue(engine.room_ok(lecture,"Theory",40))
        self.assertFalse(engine.room_ok(lab,"Theory",40))
        self.assertTrue(engine.room_ok(lab,"Lab",40))
        self.assertFalse(engine.room_ok(lecture,"Lab",40))

if __name__=="__main__":
    unittest.main()
