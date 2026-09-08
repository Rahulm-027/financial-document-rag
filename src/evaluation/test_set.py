"""
test_set.py
===========
65-query benchmark for evaluating the Financial RAG system.

Composition
-----------
  60 in-scope queries across 5 categories:
    factual       (15) — qualitative / explanatory
    numerical     (15) — specific metrics / values
    comparison    (10) — period-to-period within a company
    chart_trend   (10) — trajectories and visual/chart data
    cross_company (10) — across Tesla / Apple / Microsoft / Nvidia / Amazon

  5 out-of-scope queries (in_scope=False) — system should abstain.
  These cover companies and topics not present in the indexed corpus
  (Google, Meta, Samsung, Bitcoin, Netflix).

Schema
------
  question        : the query string
  query_type      : one of the 5 types above
  company         : primary company (or "all" for cross-company)
  year            : document year
  gold_answer     : expected answer (for LLM-judge scoring)
  evidence_pages  : list of page numbers where evidence appears
                    (page-level relevance proxy for retrieval evaluation;
                    all chunks from these pages are treated as relevant)
  in_scope        : True for normal queries, False for abstention test

IMPORTANT — gold_answer validation
-----------------------------------
  Gold answers are written as general templates based on typical 10-K/earnings
  deck content.  Before reporting evaluation results, verify each gold answer
  against your actual indexed documents and update exact figures.  The LLM
  judge compares semantic content, not exact string match, but factually
  incorrect gold answers will degrade evaluation quality.
"""

from __future__ import annotations

TEST_SET = [

    # ────────────────────────────────────────
    # FACTUAL (15 questions)
    # ────────────────────────────────────────
    {
        "question": "What were the main factors that contributed to Tesla's revenue growth in FY2023?",
        "query_type": "factual",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's revenue growth was driven by increased vehicle deliveries, expansion of Energy Generation and Storage segment, and growth in Services revenue.",
        "evidence_pages": [4, 5, 6],
        "in_scope": True,
    },
    {
        "question": "How does Tesla describe its approach to autonomous driving technology in its 10-K?",
        "query_type": "factual",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla describes Full Self-Driving as a key strategic initiative, with ongoing development of neural network-based perception and a supervised learning approach.",
        "evidence_pages": [12, 13],
        "in_scope": True,
    },
    {
        "question": "What are the primary risk factors Tesla identifies related to supply chain?",
        "query_type": "factual",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla identifies single-source suppliers, semiconductor shortages, and geographic concentration of manufacturing as key supply chain risks.",
        "evidence_pages": [18, 19, 20],
        "in_scope": True,
    },
    {
        "question": "How does Apple describe its Services segment in its 10-K?",
        "query_type": "factual",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's Services segment includes App Store, Apple Music, iCloud, Apple TV+, Apple Pay, and other subscription and licensing revenues.",
        "evidence_pages": [8, 9],
        "in_scope": True,
    },
    {
        "question": "What does Microsoft identify as the key growth drivers for its Azure cloud platform?",
        "query_type": "factual",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft cites AI integration via Azure OpenAI Service, enterprise cloud migration, and hybrid cloud solutions as primary Azure growth drivers.",
        "evidence_pages": [10, 11, 12],
        "in_scope": True,
    },
    {
        "question": "What are the major segments of Nvidia's business as described in its annual report?",
        "query_type": "factual",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia reports two primary segments: Compute & Networking (including data center GPUs) and Graphics (gaming, professional visualization).",
        "evidence_pages": [5, 6],
        "in_scope": True,
    },
    {
        "question": "How does Amazon describe the competitive landscape for AWS?",
        "query_type": "factual",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon identifies Microsoft Azure and Google Cloud as primary competitors for AWS, competing on price, breadth of services, and geographic coverage.",
        "evidence_pages": [15, 16],
        "in_scope": True,
    },
    {
        "question": "What does Tesla's 10-K say about its Gigafactory capacity expansion plans?",
        "query_type": "factual",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla describes plans to expand manufacturing capacity at Gigafactory Texas, Gigafactory Berlin, and a new facility in Mexico.",
        "evidence_pages": [22, 23],
        "in_scope": True,
    },
    {
        "question": "How does Apple discuss its exposure to foreign exchange risk?",
        "query_type": "factual",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple notes significant revenue from international markets in non-USD currencies and uses hedging instruments to manage foreign exchange exposure.",
        "evidence_pages": [30, 31],
        "in_scope": True,
    },
    {
        "question": "What strategies does Microsoft outline for its AI business?",
        "query_type": "factual",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft describes embedding Copilot AI across its product suite including Office 365, GitHub, Dynamics, and Azure as its core AI strategy.",
        "evidence_pages": [7, 8, 9],
        "in_scope": True,
    },
    {
        "question": "How does Nvidia explain the surge in data center revenue?",
        "query_type": "factual",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia attributes the data center revenue surge to strong demand for H100 GPUs from hyperscalers and large language model training workloads.",
        "evidence_pages": [8, 9, 10],
        "in_scope": True,
    },
    {
        "question": "What does Amazon say about its advertising business?",
        "query_type": "factual",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon describes advertising as one of its fastest-growing segments, primarily from sponsored product listings and display advertising.",
        "evidence_pages": [12, 13],
        "in_scope": True,
    },
    {
        "question": "What does Tesla disclose about its energy business?",
        "query_type": "factual",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's Energy Generation and Storage segment includes Powerwall, Megapack, and Solar Roof products, with Megapack deployments growing substantially.",
        "evidence_pages": [7, 8],
        "in_scope": True,
    },
    {
        "question": "How does Apple describe its approach to research and development investment?",
        "query_type": "factual",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple emphasizes ongoing heavy investment in R&D across hardware, software, and services, though it does not specify projects by name in public disclosures.",
        "evidence_pages": [25, 26],
        "in_scope": True,
    },
    {
        "question": "What regulatory risks does Nvidia highlight in its filings?",
        "query_type": "factual",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia highlights US export controls on advanced AI chips to China, antitrust scrutiny, and evolving AI regulation as key regulatory risks.",
        "evidence_pages": [20, 21, 22],
        "in_scope": True,
    },

    # ────────────────────────────────────────
    # NUMERICAL (15 questions)
    # ────────────────────────────────────────
    {
        "question": "What was Tesla's total revenue for FY2023?",
        "query_type": "numerical",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's total revenue for FY2023 was approximately $96.8 billion.",
        "evidence_pages": [4],
        "in_scope": True,
    },
    {
        "question": "What was Tesla's gross profit margin in FY2023?",
        "query_type": "numerical",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's gross profit margin for FY2023 was approximately 18.2%.",
        "evidence_pages": [4, 5],
        "in_scope": True,
    },
    {
        "question": "What was Apple's total revenue in fiscal year 2023?",
        "query_type": "numerical",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's total net sales for FY2023 were approximately $383.3 billion.",
        "evidence_pages": [30],
        "in_scope": True,
    },
    {
        "question": "What was Apple's Services segment revenue in FY2023?",
        "query_type": "numerical",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's Services revenue in FY2023 was approximately $85.2 billion.",
        "evidence_pages": [31],
        "in_scope": True,
    },
    {
        "question": "What was Microsoft's total revenue for fiscal year 2023?",
        "query_type": "numerical",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft's total revenue for FY2023 was approximately $211.9 billion.",
        "evidence_pages": [40],
        "in_scope": True,
    },
    {
        "question": "What was Microsoft's Azure revenue growth rate in FY2023?",
        "query_type": "numerical",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft reported Azure and other cloud services revenue grew approximately 27% in FY2023.",
        "evidence_pages": [41, 42],
        "in_scope": True,
    },
    {
        "question": "What was Nvidia's data center revenue in FY2024?",
        "query_type": "numerical",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia's Data Center segment revenue for FY2024 was approximately $47.5 billion.",
        "evidence_pages": [5],
        "in_scope": True,
    },
    {
        "question": "What was Nvidia's net income for FY2024?",
        "query_type": "numerical",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia reported net income of approximately $29.8 billion for FY2024.",
        "evidence_pages": [6],
        "in_scope": True,
    },
    {
        "question": "What was Amazon's AWS revenue for FY2023?",
        "query_type": "numerical",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon Web Services reported revenue of approximately $90.8 billion for FY2023.",
        "evidence_pages": [20],
        "in_scope": True,
    },
    {
        "question": "What was Amazon's operating income for FY2023?",
        "query_type": "numerical",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon's total operating income for FY2023 was approximately $36.9 billion.",
        "evidence_pages": [21],
        "in_scope": True,
    },
    {
        "question": "What were Tesla's vehicle deliveries in Q4 2023?",
        "query_type": "numerical",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla delivered approximately 484,507 vehicles in Q4 2023.",
        "evidence_pages": [3],
        "in_scope": True,
    },
    {
        "question": "What was Tesla's free cash flow in FY2023?",
        "query_type": "numerical",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla generated approximately $4.4 billion in free cash flow for FY2023.",
        "evidence_pages": [6],
        "in_scope": True,
    },
    {
        "question": "What was Apple's EPS (diluted) in FY2023?",
        "query_type": "numerical",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple reported diluted EPS of approximately $6.13 for FY2023.",
        "evidence_pages": [30],
        "in_scope": True,
    },
    {
        "question": "What was Microsoft's operating margin in FY2023?",
        "query_type": "numerical",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft's operating margin for FY2023 was approximately 41.8%.",
        "evidence_pages": [40],
        "in_scope": True,
    },
    {
        "question": "What was Nvidia's gross margin in FY2024?",
        "query_type": "numerical",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia reported a gross margin of approximately 72.7% for FY2024.",
        "evidence_pages": [5],
        "in_scope": True,
    },

    # ────────────────────────────────────────
    # COMPARISON (10 questions)
    # ────────────────────────────────────────
    {
        "question": "How did Tesla's gross margin change from FY2022 to FY2023?",
        "query_type": "comparison",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's gross margin declined from approximately 25.6% in FY2022 to approximately 18.2% in FY2023, driven primarily by price reductions.",
        "evidence_pages": [4, 5],
        "in_scope": True,
    },
    {
        "question": "How did Tesla's vehicle delivery numbers change from Q3 to Q4 2023?",
        "query_type": "comparison",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla deliveries increased from approximately 435,059 in Q3 2023 to approximately 484,507 in Q4 2023.",
        "evidence_pages": [3],
        "in_scope": True,
    },
    {
        "question": "Compare Apple's iPhone revenue in FY2022 vs FY2023.",
        "query_type": "comparison",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's iPhone revenue was approximately $205.5B in FY2022 and approximately $200.6B in FY2023, representing a modest decline.",
        "evidence_pages": [31, 32],
        "in_scope": True,
    },
    {
        "question": "How did Microsoft's Intelligent Cloud revenue growth compare between FY2022 and FY2023?",
        "query_type": "comparison",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft's Intelligent Cloud segment grew approximately 17% in FY2023, down from approximately 26% growth in FY2022.",
        "evidence_pages": [41],
        "in_scope": True,
    },
    {
        "question": "How did Nvidia's gaming revenue change from FY2023 to FY2024?",
        "query_type": "comparison",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia's Gaming segment revenue declined from approximately $9.1B in FY2023 to approximately $10.4B in FY2024 as the market recovered.",
        "evidence_pages": [7],
        "in_scope": True,
    },
    {
        "question": "How did Amazon's AWS operating margin change from 2022 to 2023?",
        "query_type": "comparison",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "AWS operating margin declined from approximately 29% in 2022 to approximately 24% in 2023, before recovering in the second half of 2023.",
        "evidence_pages": [22],
        "in_scope": True,
    },
    {
        "question": "Compare Tesla's operating expenses in H1 2023 versus H2 2023.",
        "query_type": "comparison",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's total operating expenses were slightly higher in H2 2023 vs H1 2023, driven by increased R&D spending on Cybertruck and FSD development.",
        "evidence_pages": [5, 6],
        "in_scope": True,
    },
    {
        "question": "How did Apple's Mac segment revenue compare FY2022 to FY2023?",
        "query_type": "comparison",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's Mac revenue declined from approximately $40.2B in FY2022 to approximately $29.4B in FY2023, reflecting PC market weakness.",
        "evidence_pages": [32],
        "in_scope": True,
    },
    {
        "question": "How did Microsoft's revenue growth rate change from Q1 to Q4 of FY2023?",
        "query_type": "comparison",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft's YoY revenue growth rate accelerated from approximately 2% in Q2 FY2023 to approximately 16% by Q4 FY2023.",
        "evidence_pages": [40, 41],
        "in_scope": True,
    },
    {
        "question": "How did Amazon's advertising revenue change from 2022 to 2023?",
        "query_type": "comparison",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon's advertising services revenue grew from approximately $38.0B in 2022 to approximately $46.9B in 2023, a growth rate of approximately 24%.",
        "evidence_pages": [20],
        "in_scope": True,
    },

    # ────────────────────────────────────────
    # CHART / TREND (10 questions)
    # ────────────────────────────────────────
    {
        "question": "How have Tesla's vehicle deliveries trended over the past two years?",
        "query_type": "chart_trend",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's quarterly deliveries have shown a strong upward trend, growing from around 300K per quarter in early 2022 to approximately 484K in Q4 2023.",
        "evidence_pages": [3, 4],
        "in_scope": True,
    },
    {
        "question": "Describe the trend in Nvidia's data center revenue over the past four quarters.",
        "query_type": "chart_trend",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia's data center revenue accelerated sharply from Q1 to Q3 FY2024, driven by explosive H100 GPU demand, then continued at a high level.",
        "evidence_pages": [8, 9],
        "in_scope": True,
    },
    {
        "question": "How has Apple's Services segment revenue trended since FY2019?",
        "query_type": "chart_trend",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple's Services revenue has grown consistently from approximately $46B in FY2019 to $85B+ in FY2023, roughly doubling over five years.",
        "evidence_pages": [31],
        "in_scope": True,
    },
    {
        "question": "How has Microsoft's cloud revenue mix evolved over the past three years?",
        "query_type": "chart_trend",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Cloud revenue as a share of total Microsoft revenue has grown from approximately 40% to over 50%, with Azure being the primary driver.",
        "evidence_pages": [41, 42],
        "in_scope": True,
    },
    {
        "question": "What trend does Amazon show in AWS operating income over 2022–2023?",
        "query_type": "chart_trend",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "AWS operating income declined in early 2022, was roughly flat through mid-2023, then recovered to strong growth by Q4 2023.",
        "evidence_pages": [22],
        "in_scope": True,
    },
    {
        "question": "How has Tesla's energy storage deployment (MWh) trended?",
        "query_type": "chart_trend",
        "company": "Tesla",
        "year": 2023,
        "gold_answer": "Tesla's energy storage deployments grew significantly year-over-year, with Megapack deployments in 2023 roughly tripling versus 2022 levels.",
        "evidence_pages": [7],
        "in_scope": True,
    },
    {
        "question": "How has Nvidia's gross margin trended over the last 4 quarters?",
        "query_type": "chart_trend",
        "company": "Nvidia",
        "year": 2023,
        "gold_answer": "Nvidia's gross margin expanded significantly from approximately 56% in Q1 FY2024 to over 72% by Q4 FY2024, driven by high-margin H100 GPU sales.",
        "evidence_pages": [5, 6],
        "in_scope": True,
    },
    {
        "question": "How has Apple's iPhone unit mix shifted between models over time?",
        "query_type": "chart_trend",
        "company": "Apple",
        "year": 2023,
        "gold_answer": "Apple does not disclose unit mix by model but notes the Pro lineup drives increasing revenue share due to higher average selling prices.",
        "evidence_pages": [32],
        "in_scope": True,
    },
    {
        "question": "What does the chart on Amazon's North America segment show about revenue growth?",
        "query_type": "chart_trend",
        "company": "Amazon",
        "year": 2023,
        "gold_answer": "Amazon's North America segment returned to profitability in 2023 after posting an operating loss in 2022, with revenue growing approximately 12% YoY.",
        "evidence_pages": [21],
        "in_scope": True,
    },
    {
        "question": "How has Microsoft's headcount changed relative to its revenue trend?",
        "query_type": "chart_trend",
        "company": "Microsoft",
        "year": 2023,
        "gold_answer": "Microsoft implemented layoffs in early FY2023 while revenue continued to grow, improving revenue per employee.",
        "evidence_pages": [45],
        "in_scope": True,
    },

    # ────────────────────────────────────────
    # CROSS-COMPANY (10 questions)
    # ────────────────────────────────────────
    {
        "question": "Which company had the highest revenue growth rate in FY2023: Tesla, Apple, Microsoft, Nvidia, or Amazon?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Nvidia had the highest revenue growth rate in FY2023, with total revenue growing over 120% YoY driven by AI GPU demand.",
        "evidence_pages": [],  # multiple docs
        "in_scope": True,
    },
    {
        "question": "Compare the operating margins of Apple, Microsoft, and Nvidia in their most recent fiscal year.",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Nvidia had the highest operating margin at approximately 55%, followed by Microsoft at approximately 42%, and Apple at approximately 30%.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Which company has the largest cloud business by revenue: Microsoft, Amazon, or Tesla?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Amazon's AWS had the largest cloud revenue at approximately $90.8B, followed by Microsoft's Intelligent Cloud at approximately $87.9B. Tesla does not operate a cloud business.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "How does Tesla's gross margin compare to Apple's in their most recent fiscal year?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Apple's gross margin of approximately 44% was significantly higher than Tesla's approximately 18% gross margin in their most recent fiscal years.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Which of the five companies spent the most on R&D as a percentage of revenue?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Nvidia had among the highest R&D intensity (as % of revenue), though exact comparisons require specific figures from each company's filings.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Compare Tesla and Nvidia's revenue growth rates in FY2023.",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Nvidia's revenue grew approximately 122% in FY2024 while Tesla's grew approximately 19% in FY2023, reflecting different stages of AI adoption.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Which company generated the most free cash flow in FY2023: Apple, Microsoft, or Amazon?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Apple generated the largest free cash flow at approximately $99.6B in FY2023, ahead of Microsoft at approximately $59.5B and Amazon at approximately $32.0B.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Compare the headcount and revenue per employee for Microsoft vs. Amazon.",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Microsoft had approximately 221K employees with revenue of about $212B (~$960K/employee); Amazon had approximately 1.5M employees with ~$575B revenue (~$383K/employee).",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Which company had higher EPS growth: Apple or Microsoft?",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Microsoft had higher EPS growth in FY2023, growing approximately 20% YoY vs Apple's relatively flat EPS.",
        "evidence_pages": [],
        "in_scope": True,
    },
    {
        "question": "Compare the debt-to-equity ratios of Tesla and Apple.",
        "query_type": "cross_company",
        "company": "all",
        "year": 2023,
        "gold_answer": "Apple has a negative book equity due to share buybacks, making traditional D/E comparison difficult; Tesla has manageable long-term debt relative to its equity.",
        "evidence_pages": [],
        "in_scope": True,
    },

    # ────────────────────────────────────────
    # OUT-OF-SCOPE (5 questions — system should abstain)
    # ────────────────────────────────────────
    {
        "question": "What was Google's revenue in FY2023?",
        "query_type": "factual",
        "company": "Google",
        "year": 2023,
        "gold_answer": "ABSTAIN — Google is not in the indexed document set.",
        "evidence_pages": [],
        "in_scope": False,
    },
    {
        "question": "What was Meta's net income in Q3 2023?",
        "query_type": "numerical",
        "company": "Meta",
        "year": 2023,
        "gold_answer": "ABSTAIN — Meta is not in the indexed document set.",
        "evidence_pages": [],
        "in_scope": False,
    },
    {
        "question": "How many employees does Samsung have?",
        "query_type": "factual",
        "company": "Samsung",
        "year": 2023,
        "gold_answer": "ABSTAIN — Samsung is not in the indexed document set.",
        "evidence_pages": [],
        "in_scope": False,
    },
    {
        "question": "What is the current price of Bitcoin?",
        "query_type": "factual",
        "company": "N/A",
        "year": 2023,
        "gold_answer": "ABSTAIN — No cryptocurrency price data is indexed.",
        "evidence_pages": [],
        "in_scope": False,
    },
    {
        "question": "What was Netflix's subscriber count at end of 2023?",
        "query_type": "numerical",
        "company": "Netflix",
        "year": 2023,
        "gold_answer": "ABSTAIN — Netflix is not in the indexed document set.",
        "evidence_pages": [],
        "in_scope": False,
    },
]


def get_test_set(query_type: str = None, in_scope_only: bool = False) -> list[dict]:
    """
    Return the test set, optionally filtered by query_type and/or in_scope.

    Args
    ----
    query_type    : filter to a specific type (e.g. "numerical")
    in_scope_only : if True, exclude out-of-scope questions
    """
    data = TEST_SET
    if in_scope_only:
        data = [q for q in data if q["in_scope"]]
    if query_type:
        data = [q for q in data if q["query_type"] == query_type]
    return data


def get_out_of_scope() -> list[dict]:
    return [q for q in TEST_SET if not q["in_scope"]]


if __name__ == "__main__":
    from collections import Counter
    counts = Counter(q["query_type"] for q in TEST_SET)
    total = len(TEST_SET)
    in_scope = sum(1 for q in TEST_SET if q["in_scope"])
    print(f"Total questions: {total} ({in_scope} in-scope, {total - in_scope} out-of-scope)")
    print("By type:", dict(counts))
