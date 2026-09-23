from examples.student.model import StudentSchoolModel
from mesa_llm.reasoning.react import ReActReasoning

model = StudentSchoolModel(
    width=10,
    height=10,
    n_students=10,
    n_schools=3,
    reasoning=ReActReasoning,
    llm_model="ollama/llama3.1:latest",
    vision=1,
    seed=42,
)

print("\n--- RUNNING MODEL ---")

model.step()  # only 1 step

print("\n--- RESULTS ---")

enrolled = 0
dropout = 0

for s in model.students:
    school = s.current_school.unique_id if s.current_school else None
    print(f"Student {s.unique_id}: {s.state}, school={school}")

    if s.state.value == "ENROLLED":
        enrolled += 1

    if s.state.value == "DROPOUT":
        dropout += 1

print("\nSummary:")
print("Enrolled:", enrolled)
print("Dropout:", dropout)
