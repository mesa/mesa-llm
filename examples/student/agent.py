import math
import random
from enum import Enum

from mesa_llm.llm_agent import LLMAgent
from mesa_llm.memory.st_lt_memory import STLTMemory
from mesa_llm.tools.tool_manager import ToolManager

student_tool_manager = ToolManager()
school_tool_manager = ToolManager()


class StudentState(Enum):
    ENROLLED = "ENROLLED"
    APPLIED = "APPLIED"
    DROPOUT = "DROPOUT"
    GRADUATE = "GRADUATE"


class StudentAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
        )

        self.pos = None
        self.ses = random.random()
        self.achievement = random.uniform(-1, 1)
        self.grade = random.randint(1, 12)
        self.state = StudentState.ENROLLED
        self.passed = True

        self.budget = max(
            0.0,
            model.beta_bc0
            + model.beta_bc1 * self.ses
            + random.normalvariate(0, model.sigma_bc),
        )

        self.current_school = None
        self.choice_set = []
        self.utility_scores = {}
        self.social_network = []

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            api_base=api_base,
            display=True,
        )
        self.tool_manager = student_tool_manager

    def refresh_internal_state(self):
        school_name = (
            f"School {self.current_school.unique_id}"
            if self.current_school is not None
            else "No school assigned"
        )
        friend_count = len(self.social_network)
        visible_choices = [f"School {school.unique_id}" for school in self.choice_set]
        ranked_utilities = {
            f"School {school.unique_id}": round(score, 3)
            for school, score in sorted(
                self.utility_scores.items(), key=lambda item: item[1], reverse=True
            )
        }

        self.internal_state = [
            f"My enrollment state is {self.state.value}.",
            f"My grade is {self.grade}.",
            f"My achievement score is {self.achievement:.3f}.",
            f"My socioeconomic status score is {self.ses:.3f}.",
            f"My budget is {self.budget:.2f}.",
            f"My current school is {school_name}.",
            f"My pass status is {self.passed}.",
            f"I currently track {friend_count} social connections.",
            f"My available school choices are: {visible_choices}.",
            f"My current utility ranking is: {ranked_utilities}.",
        ]

    def update_achievement(self):
        noise = random.normalvariate(0, self.model.sigma_ach)
        school_effect = self.current_school.value_added if self.current_school else 0.0

        self.achievement = (
            self.model.alpha_ach
            + self.model.beta_ach * self.achievement
            + school_effect
            + noise
        )
        self.achievement = max(-4.0, min(4.0, self.achievement))
        self.refresh_internal_state()

    def apply_dropout(self):
        if self.state in {StudentState.DROPOUT, StudentState.GRADUATE}:
            return

        if self.passed:
            probability = self.model.alpha_pass
        else:
            probability = self.model.alpha_fail * math.exp(
                self.model.beta_fail * self.grade
            )

        if random.random() < probability:
            self.state = StudentState.DROPOUT
            self.current_school = None

        self.refresh_internal_state()

    def build_social_network(self):
        self.social_network = []

        for other in self.model.students:
            if other is self or other.state == StudentState.DROPOUT:
                continue

            ses_diff = abs(self.ses - other.ses)
            probability = math.exp(self.model.theta * ses_diff) / (
                1 + math.exp(self.model.theta * ses_diff)
            )

            if random.random() < probability:
                self.social_network.append(other)

        self.refresh_internal_state()

    def build_choice_set(self):
        self.choice_set = []

        if self.pos is None:
            return

        for school in self.model.schools:
            dx = self.pos[0] - school.pos[0]
            dy = self.pos[1] - school.pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            if school.tuition > self.budget:
                continue

            if (
                distance <= self.model.distance_threshold
                or random.random() < school.visibility_prob
            ):
                self.choice_set.append(school)

        self.refresh_internal_state()

    def compute_utility(self):
        self.utility_scores = {}

        for school in self.choice_set:
            dx = self.pos[0] - school.pos[0]
            dy = self.pos[1] - school.pos[1]
            distance = math.sqrt(dx**2 + dy**2)

            social_signal = sum(
                1 for friend in self.social_network if friend.current_school == school
            )
            ses_diff = abs(self.ses - school.avg_ses)

            score = (
                self.model.beta_distance * distance
                + self.model.beta_quality * school.mean_achievement
                + self.model.beta_selectivity * int(school.selective)
                + self.model.beta_social * social_signal
                + self.model.beta_ses * ses_diff
            )
            self.utility_scores[school] = score

        self.refresh_internal_state()

    def choose_school(self):
        available_scores = {
            school: score
            for school, score in self.utility_scores.items()
            if school in self.choice_set
        }

        if not available_scores:
            return None

        schools = list(available_scores.keys())
        values = list(available_scores.values())
        max_value = max(values)
        exp_vals = [math.exp(value - max_value) for value in values]
        total = sum(exp_vals)
        probabilities = [value / total for value in exp_vals]

        return random.choices(schools, weights=probabilities, k=1)[0]

    def step(self):

        self.refresh_internal_state()

        if self.state in {StudentState.DROPOUT, StudentState.GRADUATE}:
            return

        observation = self.generate_obs()
        plan = self.reasoning.plan(
            obs=observation,
            selected_tools=["move_one_step", "speak_to", "graduate", "leave_school"],
        )
        self.apply_plan(plan)


class SchoolAgent(LLMAgent):
    def __init__(
        self,
        model,
        reasoning,
        llm_model,
        system_prompt,
        vision,
        internal_state,
        step_prompt,
        api_base=None,
    ):
        super().__init__(
            model=model,
            reasoning=reasoning,
            llm_model=llm_model,
            system_prompt=system_prompt,
            vision=vision,
            internal_state=internal_state,
            step_prompt=step_prompt,
            api_base=api_base,
        )

        self.pos = None
        self.capacity = random.randint(20, 50)
        self.students = []
        self.selective = random.choice([True, False])
        self.value_added = random.uniform(0, 1)
        self.mean_achievement = random.uniform(0, 1)
        self.visibility_prob = 0.3
        self.avg_ses = random.random()
        self.tuition = random.uniform(1, 10)

        self.memory = STLTMemory(
            agent=self,
            llm_model=llm_model,
            api_base=api_base,
            display=True,
        )
        self.tool_manager = school_tool_manager
        self.refresh_internal_state()

    def refresh_internal_state(self):
        self.internal_state = [
            f"My capacity is {self.capacity}.",
            f"My tuition is {self.tuition:.2f}.",
            f"My selectivity flag is {self.selective}.",
            f"My current enrollment is {len(self.students)} students.",
            f"My value-added effect is {self.value_added:.3f}.",
            f"My mean achievement is {self.mean_achievement:.3f}.",
            f"My average SES is {self.avg_ses:.3f}.",
        ]

    def update_tuition(self):
        noise = random.normalvariate(0, self.model.sigma_T)
        self.tuition = self.model.alpha_T + self.model.beta_T * self.tuition + noise
        self.tuition = max(0.0, self.tuition)
        self.refresh_internal_state()

    def update_composition_metrics(self):
        if self.students:
            self.mean_achievement = sum(s.achievement for s in self.students) / len(
                self.students
            )
            self.avg_ses = sum(s.ses for s in self.students) / len(self.students)
        self.refresh_internal_state()

    def select_students(self, applicants):

        remaining_capacity = self.capacity - len(self.students)
        if remaining_capacity <= 0:
            return []

        if len(applicants) <= remaining_capacity:
            return applicants

        if self.selective:
            ranked = sorted(
                applicants, key=lambda student: student.achievement, reverse=True
            )
            return ranked[: self.capacity]

        return random.sample(applicants, remaining_capacity)

    def step(self):
        self.refresh_internal_state()
        observation = self.generate_obs()
        plan = self.reasoning.plan(
            obs=observation,
            selected_tools=["speak_to"],
        )
        self.apply_plan(plan)
