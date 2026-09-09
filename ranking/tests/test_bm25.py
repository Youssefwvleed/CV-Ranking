from ranking.bm25_retriever import rank_candidates

from ranking.mock_data.mock_job_requirements import mock_job_requirements
from ranking.mock_data.mock_candidates import mock_candidates


results = rank_candidates(
    mock_job_requirements,
    mock_candidates
)

for result in results:
    print(result)