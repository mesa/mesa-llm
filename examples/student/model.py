from mesa.datacollection import DataCollector
from mesa.model import Model
from mesa.space import MultiGrid
from rich import print

from examples.student.agent import SchoolAgent, StudentAgent, StudentState
from mesa_llm.reasoning.react import ReActReasoning
from mesa_llm.reasoning.reasoning import Reasoning
from mesa_llm.recording.record_model import record_model


@record_model(output_dir="recordings")
class StudentSchoolModel(Model):
    def __init__(
        self,
        width: int,
        height: int,
        n_students: int,
        n_schools: int,
        reasoning: type[Reasoning],
        llm_model: str,
        vision: int,
        api_base: str | None = None,
        parallel_stepping=False,
        seed=None,
    ):
        super().__init__(seed=seed)
        self.width = width
        self.height = height
        self.parallel_stepping = parallel_stepping
        self.grid = MultiGrid(width, height, torus=False)

        self.students = []
        self.schools = []

        self.alpha_ach = 0.2
        self.beta_ach = 0.7
        self.sigma_ach = 0.5

        self.alpha_pass = 0.01
        self.alpha_fail = 0.05
        self.beta_fail = 0.1

        self.alpha_T = 5
        self.beta_T = 0.9
        self.sigma_T = 2

        self.beta_distance = -0.05
        self.beta_quality = 0.6
        self.beta_selectivity = 0.3
        self.beta_social = 0.4
        self.beta_ses = -0.2

        self.theta = -0.5

        self.beta_bc0 = 10
        self.beta_bc1 = 2
        self.sigma_bc = 5

        self.distance_threshold = 6
        self.pass_rate = 0.8

        self.datacollector = DataCollector(
            model_reporters={
                "Enrolled": lambda m: sum(
                    1
                    for student in m.students
                    if student.state == StudentState.ENROLLED
                ),
                "Dropout": lambda m: sum(
                    1 for student in m.students if student.state == StudentState.DROPOUT
                ),
                "Graduate": lambda m: sum(
                    1
                    for student in m.students
                    if student.state == StudentState.GRADUATE
                ),
                "Average_Achievement": lambda m: (
                    sum(student.achievement for student in m.students) / len(m.students)
                    if m.students
                    else 0.0
                ),
            },
            agent_reporters={
                "state": lambda agent: getattr(
                    getattr(agent, "state", None), "value", None
                ),
                "grade": lambda agent: getattr(agent, "grade", None),
                "achievement": lambda agent: getattr(agent, "achievement", None),
                "tuition": lambda agent: getattr(agent, "tuition", None),
            },
        )

        school_system_prompt = (
            "You are a school administrator in a competitive education market. "
            "Communicate clearly with nearby students and describe your school honestly."
        )
        school_step_prompt = (
            "Review your current capacity, tuition, and nearby students. "
            "If helpful, use speak_to to share information about your school."
        )
        schools = SchoolAgent.create_agents(
            self,
            n=n_schools,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=school_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt=school_step_prompt,
            api_base=api_base,
        )

        x_positions = self.rng.integers(0, self.grid.width, size=(n_schools,))
        y_positions = self.rng.integers(0, self.grid.height, size=(n_schools,))
        for school, x_pos, y_pos in zip(schools, x_positions, y_positions):
            self.grid.place_agent(school, (x_pos, y_pos))
            self.schools.append(school)

        student_system_prompt = (
            "You are a student navigating school choice, financial constraints, "
            "social influence, and academic progress."
        )
        student_step_prompt = (
            "Think about your academic progress, finances, and nearby schools. "
            "You may move, talk to nearby agents, graduate if you are in grade 12, "
            "or leave school if you believe continuing is impossible."
        )
        students = StudentAgent.create_agents(
            self,
            n=n_students,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=student_system_prompt,
            vision=vision,
            internal_state=None,
            step_prompt=student_step_prompt,
            api_base=api_base,
        )

        x_positions = self.rng.integers(0, self.grid.width, size=(n_students,))
        y_positions = self.rng.integers(0, self.grid.height, size=(n_students,))
        for student, x_pos, y_pos in zip(students, x_positions, y_positions):
            self.grid.place_agent(student, (x_pos, y_pos))
            self.students.append(student)

        self.refresh_all_agents()
        self.datacollector.collect(self)

    def refresh_all_agents(self):
        for school in self.schools:
            school.refresh_internal_state()
        for student in self.students:
            student.refresh_internal_state()

    def pass_fail(self):
        grades = {}

        for student in self.students:
            if student.state != StudentState.ENROLLED:
                continue
            grades.setdefault(student.grade, []).append(student)

        for grade_group in grades.values():
            ranked_students = sorted(
                grade_group,
                key=lambda student: student.achievement,
                reverse=True,
            )
            cutoff = int(len(ranked_students) * self.pass_rate)

            for index, student in enumerate(ranked_students):
                student.passed = index < cutoff
                student.refresh_internal_state()

    def matching(self):
        for school in self.schools:
            school.students = []

        for student in self.students:
            if student.state == StudentState.ENROLLED:
                student.current_school = None
                student.state = StudentState.APPLIED
                student.refresh_internal_state()

        unassigned = [
            student
            for student in self.students
            if student.state == StudentState.APPLIED
        ]

        while unassigned:
            applications = {school: [] for school in self.schools}

            for student in unassigned:
                school = student.choose_school()
                if school is not None:
                    applications[school].append(student)

            new_unassigned = []

            for school, applicants in applications.items():
                selected = school.select_students(applicants)

                for student in selected:
                    student.current_school = school
                    student.state = StudentState.ENROLLED
                    school.students.append(student)
                    student.refresh_internal_state()

                for student in applicants:
                    if student not in selected:
                        if school in student.choice_set:
                            student.choice_set.remove(school)
                        new_unassigned.append(student)
                        student.refresh_internal_state()

            unresolved_ids = {student.unique_id for student in new_unassigned}
            previous_ids = {student.unique_id for student in unassigned}
            if unresolved_ids == previous_ids:
                for student in new_unassigned:
                    student.state = StudentState.ENROLLED
                    student.refresh_internal_state()
                break

            unassigned = new_unassigned

        for student in self.students:
            if student.state == StudentState.APPLIED:
                if student.current_school is not None:
                    student.state = StudentState.ENROLLED
                else:
                    student.state = StudentState.DROPOUT
                student.refresh_internal_state()

        for school in self.schools:
            school.update_composition_metrics()

    def progress_students(self):
        for student in self.students:
            if student.state == StudentState.DROPOUT:
                continue

            if student.state == StudentState.GRADUATE:
                student.current_school = None
                student.refresh_internal_state()
                continue

            if student.passed:
                if student.grade >= 12:
                    student.state = StudentState.GRADUATE
                    student.current_school = None
                else:
                    student.grade += 1

            student.refresh_internal_state()

    def step(self):
        print(
            f"\n[bold purple] step {self.steps} "
            "────────────────────────────────────────────────────────────────────────────────[/bold purple]"
        )

        # 1. Update achievement
        for student in self.students:
            student.update_achievement()

        # 2. Build decisions
        for student in self.students:
            if student.state == StudentState.ENROLLED:
                student.build_social_network()
                student.build_choice_set()
                student.compute_utility()

        # 3. Match students to schools
        self.matching()

        # 4. Pass/fail AFTER school assignment
        self.pass_fail()

        # 5. Dropout AFTER results
        for student in self.students:
            student.apply_dropout()

        # 6. Progress grades
        self.progress_students()


if __name__ == "__main__":
    from mesa_llm.reasoning.react import ReActReasoning

    model = StudentSchoolModel(
        width=10,
        height=10,
        n_students=20,
        n_schools=4,
        reasoning=ReActReasoning,
        llm_model="ollama/llama3.1:latest",
        vision=5,
        parallel_stepping=False,
        seed=42,
    )

    for _ in range(5):
        model.step()
