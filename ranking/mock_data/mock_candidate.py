from ranking.candidate_builder import build_candidate_profile


mock_candidate_data = {
    "candidate_id": "CV_001",

    "experience_years": 4,

    "job_titles": [
        "Data Scientist",
        "Machine Learning Engineer"
    ],

    "education": [
        {
            "degree": "Bachelor",
            "field": "Computer Science"
        }
    ],

    "languages": [
        "English",
        "Arabic"
    ],

    "skills": [
        "Python",
        "SQL",
        "Pandas",
        "Machine Learning",
        "Scikit-learn"
    ],

    "certifications": []
}


mock_candidate = build_candidate_profile(mock_candidate_data)