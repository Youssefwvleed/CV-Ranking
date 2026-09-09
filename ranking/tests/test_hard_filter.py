from ranking.mock_data.mock_job_requirements import mock_job_requirements
from ranking.mock_data.mock_candidates import mock_candidates
from ranking.hard_filter import hard_filter


for candidate in mock_candidates:

    result = hard_filter(
        mock_job_requirements,
        candidate
    )

    print(
        candidate["candidate_id"],
        "→",
        result
    )