# app/data/extended_coverage_goals.py
from typing import List, Dict, Any

"""
This file defines the "Extended Coverage Goals" for the question generation system.
These goals represent strategic topics, often based on specific GRI standards,
that the system should proactively try to generate questions for, ensuring that
the platform's coverage becomes a superset of the standard SET benchmarks.

Each goal is a dictionary with the following keys:
- goal_id: A unique identifier for the goal (e.g., "EXT_GRI_302").
- goal_name: A human-readable name for the topic (e.g., "Waste Management (GRI 302)").
- goal_description: A brief explanation of what the topic covers.
- dimension: The ESG dimension ('E', 'S', or 'G').
- search_keywords: A list of keywords used by the "Guided Community Detection"
  phase to find relevant seed nodes in the knowledge graph. These keywords are
  critical for targeting the analysis.
"""

extended_coverage_goals: List[Dict[str, Any]] = [
    {
        "goal_id": "EXT_GRI_302",
        "goal_name": "Waste Management (GRI 302)",
        "goal_description": "Covers topics related to waste generation, management of significant waste-related impacts, and waste diversion.",
        "dimension": "E",
        "search_keywords": [
            "GRI 302", "waste management", "waste generation", "recycling", "recycled",
            "hazardous waste", "non-hazardous waste", "waste disposal",
            "circular economy", "waste reduction", "GRI 302-1", "GRI 302-3", "GRI 302-4"
        ]
    },
    {
        "goal_id": "EXT_GRI_303",
        "goal_name": "Water and Effluents (GRI 303)",
        "goal_description": "Addresses water as a shared resource, water consumption, discharge, and impacts on water bodies.",
        "dimension": "E",
        "search_keywords": [
            "GRI 303", "water management", "water stewardship", "water withdrawal",
            "water consumption", "water discharge", "effluents", "water stress",
            "GRI 303-1", "GRI 303-2", "GRI 303-3", "GRI 303-4", "GRI 303-5"
        ]
    },
    {
        "goal_id": "EXT_GRI_306",
        "goal_name": "Waste (2020 Standard) / Circularity (GRI 306)",
        "goal_description": "Addresses waste generation and management, focusing on preventing waste and promoting circularity.",
        "dimension": "E",
        "search_keywords": ["GRI 306", "waste", "circularity", "waste prevention", "materials recovery", "GRI 306-1", "GRI 306-2", "GRI 306-3"]
    },
    {
        "goal_id": "EXT_GRI_401",
        "goal_name": "Employment (GRI 401)",
        "goal_description": "Covers aspects like new employee hires, employee turnover, and benefits provided to full-time employees.",
        "dimension": "S",
        "search_keywords": ["GRI 401", "employment", "employee turnover", "parental leave", "benefits", "new hires", "GRI 401-1", "GRI 401-2"]
    },
    {
        "goal_id": "EXT_GRI_403",
        "goal_name": "Occupational Health and Safety (GRI 403)",
        "goal_description": "Covers occupational health and safety management systems, hazard identification, risk assessment, and incident investigation.",
        "dimension": "S",
        "search_keywords": ["GRI 403", "occupational health", "safety", "OHS", "work-related injuries", "hazard identification", "risk assessment", "GRI 403-1", "GRI 403-9"]
    },
    {
        "goal_id": "EXT_GRI_404",
        "goal_name": "Training and Education (GRI 404)",
        "goal_description": "Concerns programs for upgrading employee skills and transition assistance programs.",
        "dimension": "S",
        "search_keywords": ["GRI 404", "training", "education", "employee skills", "upskilling", "reskilling", "life-long learning", "GRI 404-1", "GRI 404-2"]
    },
    {
        "goal_id": "EXT_GRI_205",
        "goal_name": "Anti-corruption (GRI 205)",
        "goal_description": "Deals with the risks of corruption, communication and training about anti-corruption policies, and confirmed incidents.",
        "dimension": "G",
        "search_keywords": ["GRI 205", "anti-corruption", "bribery", "whistleblower", "compliance", "code of conduct", "GRI 205-1", "GRI 205-2"]
    }
]

def get_extended_coverage_goals() -> List[Dict[str, Any]]:
    return extended_coverage_goals