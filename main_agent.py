import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, TypedDict
from dataclasses import dataclass, asdict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage, SystemMessage
import json
import re
from datetime import datetime
import ast
import kagglehub
from kagglehub import KaggleDatasetAdapter

# State definition for LangGraph
class RecruitingState(TypedDict):
    job_description: str
    candidates_df: Optional[pd.DataFrame]
    assessment_criteria: Optional[Dict]
    scored_candidates: List[Dict]
    top_candidates: List[Dict]
    final_results: Optional[Dict]
    error_message: Optional[str]
    step_completed: List[str]

@dataclass
class CandidateScore:
    candidate_id: str
    name: str
    total_score: float
    skill_score: float
    experience_score: float
    education_score: float
    alignment_score: float
    detailed_evaluation: Dict
    contact_info: Dict

    def to_dict(self):
        return asdict(self)

class LangGraphRecruitingAgent:
    def __init__(self):
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash-001",
            google_api_key='api_key',
            temperature=0.1
        )

        # Create the state graph
        self.workflow = StateGraph(RecruitingState)
        self._build_graph()

    def _build_graph(self):
        """Build the LangGraph workflow"""
        # Add nodes
        self.workflow.add_node("load_dataset", self.load_dataset_node)
        self.workflow.add_node("create_criteria", self.create_criteria_node)
        self.workflow.add_node("evaluate_candidates", self.evaluate_candidates_node)
        self.workflow.add_node("rank_candidates", self.rank_candidates_node)
        self.workflow.add_node("generate_outreach", self.generate_outreach_node)
        self.workflow.add_node("compile_results", self.compile_results_node)

        # Set entry point
        self.workflow.set_entry_point("load_dataset")

        # Add edges
        self.workflow.add_edge("load_dataset", "create_criteria")
        self.workflow.add_edge("create_criteria", "evaluate_candidates")
        self.workflow.add_edge("evaluate_candidates", "rank_candidates")
        self.workflow.add_edge("rank_candidates", "generate_outreach")
        self.workflow.add_edge("generate_outreach", "compile_results")
        self.workflow.add_edge("compile_results", END)

        # Compile the graph
        self.app = self.workflow.compile()

    def load_dataset_node(self, state: RecruitingState) -> RecruitingState:
        """Load the resume dataset from Kaggle"""
        try:
            print("Loading dataset...")
            file_path = "resume_data.csv"

            # Load the latest version
            candidates_df = kagglehub.load_dataset(
                KaggleDatasetAdapter.PANDAS,
                "saugataroyarghya/resume-dataset",
                file_path,
            )
            print(f"Dataset loaded successfully. Shape: {candidates_df.shape}")

            # Select required columns
            required_columns = [
                'address', 'career_objective', 'skills', 'educational_institution_name',
                'degree_names', 'passing_years', 'educational_results', 'result_types',
                'major_field_of_studies', 'professional_company_names'
            ]

            # optional columns if they exist
            optional_columns = [
                'start_dates', 'end_dates', 'positions', 'responsibilities',
                'languages', 'certification_skills', 'experiencere_requirement'
            ]

            available_columns = [col for col in required_columns + optional_columns
                               if col in candidates_df.columns]

            candidates_df = candidates_df[available_columns]
            print(f"Selected columns: {available_columns}")

            #  Data Clean
            candidates_df = candidates_df.dropna(subset=['skills', 'career_objective'])
            print(f"After cleaning: {candidates_df.shape}")

            state["candidates_df"] = candidates_df
            state["step_completed"] = state.get("step_completed", []) + ["load_dataset"]

        except Exception as e:
            print(f"Error loading dataset: {e}")
            state["error_message"] = f"Error loading dataset: {e}"

        return state

    def create_criteria_node(self, state: RecruitingState) -> RecruitingState:
        """Create customized assessment criteria based on job description"""
        try:
            print("Creating assessment criteria...")
            job_description = state["job_description"]
            
            # Extract job keywords first
            job_keywords = self.extract_job_keywords(job_description)
            
            prompt = f"""
            Based on the following job description, create a comprehensive assessment criteria for evaluating candidates.

            Job Description:
            {job_description}

            Please provide a JSON response with the following structure:
            {{
                "technical_skills": {{
                    "required": ["skill1", "skill2", ...],
                    "preferred": ["skill1", "skill2", ...],
                    "weight": 0.4
                }},
                "experience": {{
                    "min_years": 5,
                    "relevant_domains": ["domain1", "domain2", ...],
                    "weight": 0.3
                }},
                "education": {{
                    "required_degree": "Bachelor's",
                    "preferred_fields": ["field1", "field2", ...],
                    "weight": 0.2
                }},
                "soft_skills": {{
                    "required": ["skill1", "skill2", ...],
                    "weight": 0.1
                }}
            }}

            Focus on extracting specific technical skills, experience requirements, and educational qualifications mentioned in the job description.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            try:
                # Extract JSON from response
                json_str = response.content.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:-3]
                elif json_str.startswith('```'):
                    json_str = json_str[3:-3]

                criteria = json.loads(json_str)
                # Add job keywords to criteria
                criteria["job_keywords"] = job_keywords
                state["assessment_criteria"] = criteria
                state["step_completed"] = state.get("step_completed", []) + ["create_criteria"]
                print("Assessment criteria created successfully")

            except Exception as e:
                print(f"Error parsing assessment criteria: {e}")
                # Return default criteria with job keywords
                state["assessment_criteria"] = {
                    "technical_skills": {
                        "required": ["Java", "Python", "C++", "SQL", "HTML5", "CSS3"],
                        "preferred": ["Spring MVC", "GWT", "Wicket", "ORM"],
                        "weight": 0.4
                    },
                    "experience": {
                        "min_years": 5,
                        "relevant_domains": ["software engineering", "web development", "networking"],
                        "weight": 0.3
                    },
                    "education": {
                        "required_degree": "Bachelor's",
                        "preferred_fields": ["software engineering", "computer science", "information technology"],
                        "weight": 0.2
                    },
                    "soft_skills": {
                        "required": ["problem solving", "teamwork", "communication"],
                        "weight": 0.1
                    },
                    "job_keywords": job_keywords
                }
                state["step_completed"] = state.get("step_completed", []) + ["create_criteria"]

        except Exception as e:
            print(f"Error creating criteria: {e}")
            state["error_message"] = f"Error creating criteria: {e}"

        return state

    def evaluate_candidates_node(self, state: RecruitingState) -> RecruitingState:
        """Evaluate all candidates against the job requirements"""
        try:
            print("Evaluating candidates...")
            candidates_df = state["candidates_df"]
            criteria = state["assessment_criteria"]

            if candidates_df is None or criteria is None:
                state["error_message"] = "Dataset or criteria not available"
                return state

            print(f"Evaluating {len(candidates_df)} candidates...")

            scored_candidates = []
            for idx, candidate in candidates_df.iterrows():
                try:
                    score = self.evaluate_candidate(candidate, criteria)
                    scored_candidates.append(score.to_dict())
                except Exception as e:
                    print(f"Error evaluating candidate {idx}: {e}")
                    continue

            state["scored_candidates"] = scored_candidates
            state["step_completed"] = state.get("step_completed", []) + ["evaluate_candidates"]
            print(f"Evaluation completed. {len(scored_candidates)} candidates scored.")

        except Exception as e:
            print(f"Error evaluating candidates: {e}")
            state["error_message"] = f"Error evaluating candidates: {e}"

        return state

    def rank_candidates_node(self, state: RecruitingState) -> RecruitingState:
        """Rank candidates and select top 3"""
        try:
            print("Ranking candidates...")
            scored_candidates = state["scored_candidates"]

            if not scored_candidates:
                state["error_message"] = "No scored candidates available"
                return state

            # Sort by total score
            sorted_candidates = sorted(scored_candidates, key=lambda x: x["total_score"], reverse=True)

            # Get top 3
            top_candidates = sorted_candidates[:3]
            state["top_candidates"] = top_candidates
            state["step_completed"] = state.get("step_completed", []) + ["rank_candidates"]
            print(f"Top 3 candidates selected")

        except Exception as e:
            print(f"Error ranking candidates: {e}")
            state["error_message"] = f"Error ranking candidates: {e}"

        return state

    def generate_outreach_node(self, state: RecruitingState) -> RecruitingState:
        """Generate outreach emails and explanations for top candidates"""
        try:
            print("Generating outreach content...")
            top_candidates = state["top_candidates"]

            if not top_candidates:
                state["error_message"] = "No top candidates available"
                return state

            # Generate outreach emails and explanations
            for i, candidate in enumerate(top_candidates):
                candidate["outreach_email"] = self.generate_outreach_email(candidate)
                candidate["decision_explanation"] = self.explain_decision(candidate)
                candidate["rank"] = i + 1

            state["top_candidates"] = top_candidates
            state["step_completed"] = state.get("step_completed", []) + ["generate_outreach"]
            print("Outreach content generated")

        except Exception as e:
            print(f"Error generating outreach: {e}")
            state["error_message"] = f"Error generating outreach: {e}"

        return state

    def compile_results_node(self, state: RecruitingState) -> RecruitingState:
        """Compile final results"""
        try:
            print("Compiling final results...")

            results = {
                'top_candidates': state["top_candidates"],
                'assessment_criteria': state["assessment_criteria"],
                'total_candidates_evaluated': len(state["scored_candidates"]),
                'steps_completed': state.get("step_completed", []),
                'error_message': state.get("error_message")
            }

            state["final_results"] = results
            state["step_completed"] = state.get("step_completed", []) + ["compile_results"]
            print("Results compiled successfully")

        except Exception as e:
            print(f"Error compiling results: {e}")
            state["error_message"] = f"Error compiling results: {e}"

        return state

    def extract_skills_from_text(self, text: str) -> List[str]:
        """Extract skills from text"""

        if pd.isna(text) or not text:
            return []
        
        # If it's already a list
        if isinstance(text, list):
            return [skill.strip() for skill in text]
        
        # If it's a string representation of a list
        if isinstance(text, str):
            # Try to parse as Python list
            try:
                skills_list = ast.literal_eval(text)
                if isinstance(skills_list, list):
                    return [skill.strip() for skill in skills_list]
            except:
                # If that fails, try splitting by common delimiters
                if ',' in text:
                    return [skill.strip() for skill in text.split(',')]
        
        return []
        # return found_skills

    def calculate_experience_years(self, start_dates: str, end_dates: str) -> float:
        """Calculate years of experience from start and end dates"""
        try:
            if pd.isna(start_dates) or pd.isna(end_dates):
                return 0.0

            start_years = str(start_dates).split(',') if ',' in str(start_dates) else [str(start_dates)]
            end_years = str(end_dates).split(',') if ',' in str(end_dates) else [str(end_dates)]

            total_years = 0
            for i, start_year in enumerate(start_years):
                try:
                    start = int(re.findall(r'\d{4}', start_year)[0]) if re.findall(r'\d{4}', start_year) else 2020
                    end = int(re.findall(r'\d{4}', end_years[i])[0]) if i < len(end_years) and re.findall(r'\d{4}', end_years[i]) else 2024
                    total_years += max(0, end - start)
                except:
                    continue

            return total_years
        except:
            return 0.0

    def evaluate_candidate(self, candidate_row: pd.Series, criteria: Dict) -> CandidateScore:
        """Evaluate a single candidate against the criteria"""
        
        # Extract candidate information
        candidate_skills = self.extract_skills_from_text(candidate_row.get('skills', ''))
        career_objective = str(candidate_row.get('career_objective', ''))
        education = str(candidate_row.get('degree_names', ''))
        education_results = str(candidate_row.get('educational_results', ''))

        # Calculate experience years
        experience_years = self.calculate_experience_years(
            candidate_row.get('start_dates', ''),
            candidate_row.get('end_dates', '')
        )

        # Score technical skills 40% weight
        required_skills = criteria['technical_skills']['required']
        preferred_skills = criteria['technical_skills']['preferred']

        required_matches = sum(1 for skill in required_skills if skill.lower() in [s.lower() for s in candidate_skills])
        preferred_matches = sum(1 for skill in preferred_skills if skill.lower() in [s.lower() for s in candidate_skills])

        skill_score = (required_matches / len(required_skills)) * 0.7 + (preferred_matches / len(preferred_skills)) * 0.3
        skill_score = min(skill_score, 1.0)

        # Score experience 30% weight
        min_years = criteria['experience']['min_years']
        experience_score = min(experience_years / min_years, 1.0) if min_years > 0 else 0.5

        # Score education 20% weight
        education_score = 0.5  # Default score
        if any(degree in education.lower() for degree in ['bachelor', 'master', 'phd']):
            education_score = 0.8
        if any(field in education.lower() for field in criteria['education']['preferred_fields']):
            education_score = min(education_score + 0.2, 1.0)

        # Score alignment with career objective using dynamic keywords 10% weight
        job_keywords = criteria.get('job_keywords', {})
        alignment_score = self.calculate_alignment_score(career_objective, criteria, job_keywords)

        # Calculate total score
        total_score = (
            skill_score * criteria['technical_skills']['weight'] +
            experience_score * criteria['experience']['weight'] +
            education_score * criteria['education']['weight'] +
            alignment_score * criteria['soft_skills']['weight']
        )

        # Create detailed evaluation
        detailed_evaluation = {
            'matched_skills': [skill for skill in candidate_skills if skill.lower() in [s.lower() for s in required_skills + preferred_skills]],
            'missing_skills': [skill for skill in required_skills if skill.lower() not in [s.lower() for s in candidate_skills]],
            'experience_years': experience_years,
            'education_level': education,
            'career_alignment': alignment_score,
            'alignment_keywords_matched': self._get_matched_keywords(career_objective, job_keywords)
        }

        return CandidateScore(
            candidate_id=str(candidate_row.name),
            name=f"Candidate_{candidate_row.name}",
            total_score=total_score * 100,
            skill_score=skill_score * 100,
            experience_score=experience_score * 100,
            education_score=education_score * 100,
            alignment_score=alignment_score * 100,
            detailed_evaluation=detailed_evaluation,
            contact_info={'address': candidate_row.get('address', 'N/A')}
        )

    def _get_matched_keywords(self, career_objective: str, job_keywords: Dict) -> Dict:
        """Helper method to get matched keywords for detailed evaluation"""
        if not career_objective or pd.isna(career_objective):
            return {}
        
        objective_lower = str(career_objective).lower()
        matched = {}
        
        for category, keywords in job_keywords.items():
            matched[category] = [keyword for keyword in keywords if keyword.lower() in objective_lower]
        
        return matched

    def extract_job_keywords(self, job_description: str) -> Dict:
        """Extract relevant keywords from job description using LLM"""
        try:
            prompt = f"""
            Analyze the following job description and extract key terms that would be relevant for matching candidate career objectives and skills.

            Job Description:
            {job_description}

            Please provide a JSON response with the following structure:
            {{
                "role_keywords": ["keyword1", "keyword2", ...],
                "technical_keywords": ["tech1", "tech2", ...],
                "domain_keywords": ["domain1", "domain2", ...],
                "action_keywords": ["action1", "action2", ...]
            }}

            Where:
            - role_keywords: Job titles, positions, roles mentioned
            - technical_keywords: Technologies, tools, programming languages, frameworks
            - domain_keywords: Industry domains, business areas, specializations
            - action_keywords: Action verbs describing what the role involves (develop, design, manage, etc.)

            Extract only the most relevant terms. Keep each category to 5-10 items maximum.
            """

            response = self.llm.invoke([HumanMessage(content=prompt)])
            
            try:
                json_str = response.content.strip()
                if json_str.startswith('```json'):
                    json_str = json_str[7:-3]
                elif json_str.startswith('```'):
                    json_str = json_str[3:-3]
                
                keywords = json.loads(json_str)
                return keywords
            except:
                # Fallback to default keywords if parsing fails
                return {
                    "role_keywords": ["engineer", "developer", "software", "technical"],
                    "technical_keywords": ["programming", "development", "coding", "technology"],
                    "domain_keywords": ["software", "web", "application", "system"],
                    "action_keywords": ["develop", "build", "create", "design"]
                }
        except Exception as e:
            print(f"Error extracting keywords: {e}")
            return {
                "role_keywords": ["engineer", "developer", "software", "technical"],
                "technical_keywords": ["programming", "development", "coding", "technology"],
                "domain_keywords": ["software", "web", "application", "system"],
                "action_keywords": ["develop", "build", "create", "design"]
            }

    def calculate_alignment_score(self, career_objective: str, criteria: Dict, job_keywords: Dict = None) -> float:
        """Calculate how well career objective aligns with job requirements using dynamic keywords"""
        if not career_objective or pd.isna(career_objective):
            return 0.3

        objective_lower = str(career_objective).lower()
        
        # If job_keywords not provided, use basic scoring
        if not job_keywords:
            return 0.5
        
        # Calculate matches for each keyword category
        role_matches = sum(1 for keyword in job_keywords.get('role_keywords', []) 
                        if keyword.lower() in objective_lower)
        tech_matches = sum(1 for keyword in job_keywords.get('technical_keywords', []) 
                        if keyword.lower() in objective_lower)
        domain_matches = sum(1 for keyword in job_keywords.get('domain_keywords', []) 
                            if keyword.lower() in objective_lower)
        action_matches = sum(1 for keyword in job_keywords.get('action_keywords', []) 
                            if keyword.lower() in objective_lower)
        
        # Calculate weighted score
        total_possible = (len(job_keywords.get('role_keywords', [])) + 
                        len(job_keywords.get('technical_keywords', [])) + 
                        len(job_keywords.get('domain_keywords', [])) + 
                        len(job_keywords.get('action_keywords', [])))
        
        if total_possible == 0:
            return 0.5
        
        total_matches = role_matches + tech_matches + domain_matches + action_matches
        
        # Apply different weights to different categories
        weighted_score = (
            (role_matches / max(len(job_keywords.get('role_keywords', [])), 1)) * 0.3 +
            (tech_matches / max(len(job_keywords.get('technical_keywords', [])), 1)) * 0.4 +
            (domain_matches / max(len(job_keywords.get('domain_keywords', [])), 1)) * 0.2 +
            (action_matches / max(len(job_keywords.get('action_keywords', [])), 1)) * 0.1
        )
        
        return min(weighted_score, 1.0)

    def generate_outreach_email(self, candidate: Dict) -> str:
        """Generate personalized outreach email for a candidate"""

        matched_skills = candidate['detailed_evaluation']['matched_skills']
        experience_years = candidate['detailed_evaluation']['experience_years']
        education_level = candidate['detailed_evaluation']['education_level']

        prompt = f"""
        Generate a professional and personalized outreach email for the following candidate:

        Candidate Details:
        - Name: {candidate['name']}
        - Total Score: {candidate['total_score']:.1f}%
        - Matched Skills: {', '.join(matched_skills)}
        - Experience Years: {experience_years}
        - Education: {education_level}

        Job Role: Software Engineer

        The email should:
        1. Be professional and engaging
        2. Highlight their relevant skills and experience
        3. Explain why they're a good fit
        4. Include a clear call to action for an interview
        5. Be personalized based on their background

        Keep it concise (200-300 words) and professional.
        """

        response = self.llm.invoke([HumanMessage(content=prompt)])
        return response.content

    def explain_decision(self, candidate: Dict) -> str:
        """Explain why this candidate was selected"""

        matched_skills = candidate['detailed_evaluation']['matched_skills']
        missing_skills = candidate['detailed_evaluation']['missing_skills']
        experience_years = candidate['detailed_evaluation']['experience_years']
        education_level = candidate['detailed_evaluation']['education_level']

        explanation = f"""
        Decision Explanation for {candidate['name']}:

        Overall Score: {candidate['total_score']:.1f}%

        Breakdown:
        - Technical Skills: {candidate['skill_score']:.1f}%
        - Experience: {candidate['experience_score']:.1f}%
        - Education: {candidate['education_score']:.1f}%
        - Alignment: {candidate['alignment_score']:.1f}%

        Key Strengths:
        - Matched Skills: {', '.join(matched_skills) if matched_skills else 'Basic technical foundation'}
        - Experience: {experience_years} years
        - Education: {education_level}

        Areas for Development:
        - Missing Skills: {', '.join(missing_skills) if missing_skills else 'None identified'}

        This candidate ranks in the top tier due to their strong technical foundation and relevant experience.
        """

        return explanation

    def run_evaluation(self, job_description: str) -> Dict:
        """Run the complete evaluation process using LangGraph"""
        print("Starting LangGraph evaluation process...")

        # Initial state
        initial_state = {
            "job_description": job_description,
            "candidates_df": None,
            "assessment_criteria": None,
            "scored_candidates": [],
            "top_candidates": [],
            "final_results": None,
            "error_message": None,
            "step_completed": []
        }

        # Run the workflow
        try:
            final_state = self.app.invoke(initial_state)

            if final_state.get("error_message"):
                print(f"Error during execution: {final_state['error_message']}")
                return {"error": final_state["error_message"]}

            return final_state["final_results"]

        except Exception as e:
            print(f"Error running workflow: {e}")
            return {"error": str(e)}

# Testing Module
class LangGraphRecruitingAgentTester:
    def __init__(self, agent: LangGraphRecruitingAgent):
        self.agent = agent

    def test_skill_extraction(self):
        """Test skill extraction functionality"""
        print("Testing skill extraction...")

        test_skills = "['Big Data', 'Hadoop', 'Hive', 'Python', 'Mapreduce', 'Spark', 'Java', 'Machine Learning', 'Cloud', 'Hdfs', 'YARN', 'Core Java', 'Data Science', 'C++', 'Data Structures', 'DBMS', 'RDBMS', 'Informatica', 'Talend', 'Amazon Redshift', 'Microsoft Azure']"

        print(" test_skills found ",self.agent.extract_skills_from_text(test_skills))

    def test_experience_calculation(self):
        """Test experience calculation"""
        print("Testing experience calculation...")

        test_cases = [
            ("2020,2022", "2022,2024"),
            ("2018", "2023"),
            ("", ""),
            ("2019,2021", "2021,2023")
        ]

        for i, (start, end) in enumerate(test_cases):
            years = self.agent.calculate_experience_years(start, end)
            print(f"Test {i+1}: {start} -> {end} = {years} years")

    def test_graph_structure(self):
        """Test the LangGraph structure"""
        print("Testing LangGraph structure...")

        # Test that the graph was built correctly
        print(f"Graph nodes: {list(self.agent.workflow.nodes.keys())}")
        print(f"Graph edges: {list(self.agent.workflow.edges)}")
        print("Graph structure is valid")

    def run_all_tests(self):
        """Run all tests"""
        print("=== Running LangGraph Recruiting Agent Tests ===")
        self.test_skill_extraction()
        print()
        self.test_experience_calculation()
        print()
        self.test_graph_structure()
        print("=== Tests Completed ===")

# Usage Example and Main Execution
def main():
    # Software Engineer Job Description
    job_description = """
    At [Company X], our technology solves problems. We've established the company as a leading developer of innovative software solutions, and we're looking for a highly skilled software engineer to join our program and network design team. The ideal candidate will have expert knowledge of software development processes, along with solid experience in testing and evaluating current networking systems.

    Objectives of this role:
    • Enhance existing platform and network capabilities to handle massive growth
    • Visualize, design, and develop innovative software platforms
    • Create scalable software platforms and applications with unit testing and code review
    • Examine existing systems for flaws and create solutions

    Responsibilities:
    • Design and build tools and frameworks to automate development, testing, deployment
    • Plan and scale distributed software and applications
    • Collaborate with global team to produce project plans
    • Track, document, and maintain software and network system functionality

    Required skills and qualifications:
    • Five or more years of experience as engineer of software and networking platforms
    • Seven or more years of experience with Java, Python, and C++
    • Proven ability to document design processes
    • Experience with rapid development cycles in a web-based environment
    • Strong ability in scripting and test automation

    Preferred skills and qualifications:
    • Bachelor's degree in software engineering or information technology
    • Working knowledge of relational databases as well as ORM and SQL technologies
    • Proficiency with HTML5, CSS3, and content management systems
    • Web application development experience with multiple frameworks, including Wicket, GWT, and Spring MVC
    """

    # job_description = """
    # At [Company Y], we are building the next generation of cloud-native applications to solve real-world business problems. As part of our growing engineering team, we're looking for a passionate and experienced software engineer who thrives in a fast-paced, collaborative environment. The ideal candidate is a problem-solver with deep expertise in cloud development, DevOps practices, and scalable web application design.

    # Objectives of this role:
    # • Design and develop scalable, cloud-native applications with high availability
    # • Automate infrastructure and deployment pipelines using modern DevOps tools
    # • Collaborate with cross-functional teams to translate business needs into technical solutions
    # • Champion software engineering best practices and mentor junior developers

    # Responsibilities:
    # • Build backend services and APIs using Node.js, TypeScript, or Go
    # • Design and manage cloud infrastructure (AWS / Azure / GCP) using Terraform or similar IaC tools
    # • Implement CI/CD pipelines with GitHub Actions, Jenkins, or similar tools
    # • Monitor and troubleshoot production systems using tools like Prometheus, Grafana, and ELK stack
    # • Write unit and integration tests, conduct code reviews, and ensure code quality

    # Required skills and qualifications:
    # • 4+ years of experience in full-stack or backend software development
    # • Proficiency with JavaScript/TypeScript, Node.js, and RESTful API design
    # • Experience with cloud platforms (AWS, Azure, or GCP)
    # • Hands-on experience with Docker and Kubernetes in production environments
    # • Familiarity with CI/CD tools and automated testing frameworks

    # Preferred skills and qualifications:
    # • Bachelor’s degree in Computer Science, Engineering, or related field
    # • Knowledge of serverless architectures (e.g., AWS Lambda, Azure Functions)
    # • Familiarity with frontend frameworks such as React or Angular
    # • Experience with databases like PostgreSQL, MongoDB, or DynamoDB
    # • Understanding of security best practices and API authentication protocols (OAuth2, JWT)
    # """

    # job_description = """
    # At [Company Z], we empower data-driven decision-making across industries. We’re seeking a skilled data engineer to join our growing data platform team. The ideal candidate will have a strong background in building data pipelines, handling large-scale data processing, and optimizing data workflows in cloud environments.

    # Objectives of this role:
    # • Build and maintain robust data pipelines to support analytics and ML initiatives
    # • Develop data models and implement ETL processes across distributed systems
    # • Ensure high data quality, reliability, and availability across the platform

    # Responsibilities:
    # • Design and develop scalable ETL/ELT pipelines using Apache Airflow, Spark, or dbt
    # • Integrate data from multiple sources into a centralized data lake or warehouse
    # • Collaborate with data scientists and analysts to optimize data access patterns
    # • Maintain data infrastructure in cloud platforms like AWS (Glue, Redshift, S3) or GCP (BigQuery, Dataflow)

    # Required skills and qualifications:
    # • 3+ years of experience in data engineering or backend data systems
    # • Proficiency in Python or Scala for data processing
    # • Experience with SQL and NoSQL databases
    # • Familiarity with distributed data systems like Hadoop, Spark, or Kafka
    # • Strong understanding of data warehouse/lake architecture

    # Preferred skills and qualifications:
    # • Bachelor's degree in Computer Science, Data Engineering, or similar
    # • Hands-on experience with tools like dbt, Airflow, or Dagster
    # • Exposure to data governance, lineage, and cataloging tools (e.g., Apache Atlas, Amundsen)
    # """

    # Initialize the LangGraph recruiting agent
    agent = LangGraphRecruitingAgent()

    # Run tests
    tester = LangGraphRecruitingAgentTester(agent)
    tester.run_all_tests()

    print("\n" + "="*50)
    print("LANGGRAPH RECRUITING AGENT EVALUATION")
    print("="*50)

    try:
        # Run complete evaluation using LangGraph
        results = agent.run_evaluation(job_description)

        if "error" in results:
            print(f"Error: {results['error']}")
            return

        # Display results
        print(f"\nEvaluation Results:")
        print(f"Total candidates evaluated: {results['total_candidates_evaluated']}")
        print(f"Steps completed: {results['steps_completed']}")
        print(f"Assessment criteria applied: {json.dumps(results['assessment_criteria'], indent=2)}")

        print("\n" + "="*30)
        print("TOP 3 CANDIDATES")
        print("="*30)

        for candidate in results['top_candidates']:
            print(f"\nRANK {candidate['rank']}: {candidate['name']}")
            print(f"Score: {candidate['total_score']:.1f}%")
            print(f"Contact: {candidate['contact_info']['address']}")

            print("\nDetailed Scores:")
            print(f"  Technical Skills: {candidate['skill_score']:.1f}%")
            print(f"  Experience: {candidate['experience_score']:.1f}%")
            print(f"  Education: {candidate['education_score']:.1f}%")
            print(f"  Alignment: {candidate['alignment_score']:.1f}%")

            print("\nKey Strengths:")
            print(f"  Matched Skills: {', '.join(candidate['detailed_evaluation']['matched_skills'])}")
            print(f"  Experience: {candidate['detailed_evaluation']['experience_years']} years")

            print("\nOutreach Email:")
            print(candidate['outreach_email'])

            print("\nDecision Explanation:")
            print(candidate['decision_explanation'])

            print("-" * 50)

    except Exception as e:
        print(f"Error during execution: {e}")
        print("Please ensure you have the dataset file available and Google API key configured.")

# Scaled Evaluation Approach
class ScaledEvaluationSuggestions:
    """
    Suggestions for scaling the LangGraph recruiting agent evaluation pipeline
    """

    @staticmethod
    def print_scaling_suggestions():
        print("\n" + "="*50)
        print("LANGGRAPH SCALED EVALUATION APPROACH")
        print("="*50)

        suggestions = """
            1. PARALLEL PROCESSING WITH LANGGRAPH:
                - LangGraph's parallel execution nodes for simultaneous candidate evaluation
                - Implement batch processing nodes for large datasets
                - Process multiple candidates concurrently, reducing evaluation time from hours to minutes
            
            2. DISTRIBUTED WORKFLOW:
                - Deploy multiple LangGraph instances across different servers/containers
                - Use message queues (Redis/RabbitMQ) for coordination between instances
                - Implement checkpointing for fault tolerance and resume capabilities
            
            3. CACHING & OPTIMIZATION:
                - Cache LLM responses to reduce API calls and improve response times
                - Use vector databases (Pinecone/Weaviate) for efficient skill matching and similarity search
                - Implement result caching for candidate evaluations and rankings
            
            4. REAL-TIME PROCESSING:
                - Process candidates as they're uploaded rather than batch processing
                - Use WebSocket connections for real-time status updates to users
                - Implement event-driven architecture for automatic evaluation triggers
            
            5. DATABASE & STORAGE:
                - Replace in-memory storage with PostgreSQL or MongoDB for scalability
                - Implement proper indexing on candidate skills, experience, and evaluation scores
            
            6. PERFORMANCE MONITORING & OBSERVABILITY:
                - Track processing time, accuracy rates, and system performance metrics
                - Monitor each pipeline stage to identify and resolve performance bottlenecks
                - Implement automated health checks for system reliability and uptime
        """

        print(suggestions)

if __name__ == "__main__":
    main()
    ScaledEvaluationSuggestions.print_scaling_suggestions()
