import json
import time
import aiohttp

import wikipediaapi 
import wikipedia as wpsearch # Older library that supports search but borks other stuff
from llm import get_response_openai_nonstream, OpenAifunction, OpenaiChatMessage
    


wiki = wikipediaapi.Wikipedia(
 user_agent="OpenAI Wikipedia/0.1",
language="en",
)


async def fetch_relevant_wikipedia_pages(
    topic: str, session: aiohttp.ClientSession
) -> list[wikipediaapi.WikipediaPage]:
    """
    Fetches the Wikipedia page for the given topic.

    Args:
        topic (str): Topic of interest.
        session (aiohttp.ClientSession): The aiohttp client session for making requests.

    Returns:
        wikipediaapi.WikipediaPage: The Wikipedia page.
    """

    candidates = wpsearch.search(topic, results = 5)
    pages: list[wikipediaapi.WikipediaPage] = []
    for candidate in candidates:
        page = wiki.page(candidate)
        if not page.exists():
            continue
        pages.append(page)

    return pages


def extract_sections(
    page: wikipediaapi.WikipediaPage| wikipediaapi.WikipediaPageSection, min_words: int = 250, max_wrds: int = 1000
) -> list[tuple[str, str]]:
    """
    Extracts sections from the Wikipedia page that are at least min_length words long.

    Args:
        page (wikipediaapi.WikipediaPage): The Wikipedia page.
        min_length (int, optional): Minimum number of words in a section. Defaults to 400.

    Returns:
        List[str]: A list of extracted sections from the page.
    """

    results: list[tuple[str,str]] = []
    for section in page.sections:
        if len(section.text.split()) > min_words and len(section.text.split()) < max_wrds:
            results.append((section.text, section.title))

        for subsection in section.sections:
            results += extract_sections(subsection, min_words, max_wrds)
        

    return results
        

async def rank_section(section: str, title: str,   topic: str) -> dict:

    system_prompt = """
    You are an educational expert who is currently evaluating texts to be used for assessing student skills on the CCSS.ELA-Literacy.W.4 common core standard. The standard is: 

"Draw evidence from literary or informational texts to support analysis, reflection, and research."

You score each text on a scale from 1 to 5 along the following criteria:

- Relevance to the topic: How relevant is the text to the topic? Does it directly address the topic or is it only tangentially related? (1 = not relevant, 5 = very relevant)
- Age-appropriateness: Is the text appropriate for 4th graders? Are the subjects and topics approached appropriate for children of that age (focus on the topics themselves, not on the complexity. E.g. does it talk about sexual or other adult topics)? (1 = not appropriate, 5 = very appropriate)
- Complexity fit: Does the text have the right level of complexity for 4th graders? Is the text too simple (in which case it would not be challenging enough) or too complex (in which case it would be too challenging)? Consider both vocabulary, syntax, and contents (1 = not appropriate, 5 = very appropriate)
- Potential for assessment: How well does the text lend itself to assess the standard? Does it contain information that would allow asking questions that let the student demonstrate their ability to draw evidence from the text and reflect on it? (1 = not appropriate, 5 = very appropriate)
- Overall educational value: overall, how "interesting" is the text"? Does it broach important social or scientific topics? Or is it a very niche or technical text? We are not interested in simple enumerations of facts. (1 = not interesting, 5 = very interesting)


When you receive a text and a topic, you evaluate it to see if it is appropriate for assessing this standard. To do so, you write brief reasoning (no more than two sentences) about your thoughts for each criterium containint at least one positive AND one negative point. Then give a numerical score for that criterium. Afterwards, you use the function `add_assessment` to save your evaluation of the text.
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=system_prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""
TOPIC: {topic}

====================

TEXT: 

# {title}

{section}
""",
        ),
    ]

    add_assessment_openai_function: OpenAifunction = {
        "name": "add_assessment",
        "description": "Add an assessment for the given text and topic.",
        "parameters": {
            "type": "object",
            "properties": {
                "relevance_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the relevance score. Includes at least one positive and one negative point.",
                },
                "relevance_score": {
                    "type": "number",
                    "description": "Your relevance score.",
                },
                "age_appropriateness_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the age-appropriateness score. Includes at least one positive and one negative point.",
                },
                "age_appropriateness_score": {
                    "type": "number",
                    "description": "Your age-appropriateness score.",
                },
                "complexity_fit_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the complexity fit score. Includes at least one positive and one negative point.",
                },
                "complexity_fit_score": {
                    "type": "number",
                    "description": "Your complexity fit score.",
                },
                "potential_for_assessment_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the potential for assessment score. Includes at least one positive and one negative point.",
                },
                "potential_for_assessment_score": {
                    "type": "number",
                    "description": "Your potential for assessment score.",
                },
                "overall_educational_value_reasoning": {
                    "type": "string",
                    "description": "Your reasoning for the overall educational value score. Includes at least one positive and one negative point.",
                },
                "overall_educational_value_score": {
                    "type": "number",
                    "description": "Your overall educational value score.",
                },

            },
            "required": [
                "relevance_reasoning",
                "relevance_score",
                "age_appropriateness_reasoning",
                "age_appropriateness_score",
                "complexity_fit_reasoning",
                "complexity_fit_score",
                "potential_for_assessment_reasoning",
                "potential_for_assessment_score",
                "overall_educational_value_reasoning",
                "overall_educational_value_score",

            ],
        },
    }
    
    print("Sending to OpenAI")
    start_time = time.time()
    arguments = await get_response_openai_nonstream(
        messages_for_openai, functions=[add_assessment_openai_function], function_name="add_assessment"
    )
    print(f"OpenAI response time: {time.time() - start_time}")
    # arguments = json.loads(response["arguments"])
    # arguments =/
    arguments["text"] = section
    arguments["title"] = title
    return arguments


def select_best_text(text_rankings: list[dict]) -> dict:
    
    # Start by removing any text for which age-appropriateness is below 3
    text_rankings = [text for text in text_rankings if text["age_appropriateness_score"] > 3]


    # Compute average score for each text
    for text in text_rankings:
        text["average_score"] = (
            text["relevance_score"]
            + text["age_appropriateness_score"]
            + text["complexity_fit_score"]
            + text["potential_for_assessment_score"]
            + text["overall_educational_value_score"]
        ) / 5

    # Sort by average score
    text_rankings.sort(key=lambda x: x["average_score"], reverse=True)

    # Return the top text
    return text_rankings[0]

async def clean_and_format_text(text: str) -> str:
    # prompt = f"""
    # You are tasked with cleaning up texts so that they are easily readable and well formatted. Given a text, you do the following tasks:

    # - Format them as valid markdown, escaping any special characters.
    # - Remove any HTML tags and replace them with markdown equivalents when possible
    # - Fix spacing issues (e.g. double spaces, etc.) and remove weird characters (replace them by unicode when possible)
    # - Remove any text that is not part of the main body of the text (e.g. references, etc.)

    # Given a text, you respond ONLY with the reformatted text. You do not need to provide any reasoning. You do not add any headers or footers. 
    # Make sure to preserve all existing formatting (lists, headers etc.).
    # """

    prompt = f"""
You are tasked with rewriting a text in order to make it accessible to a 4th grader.
Given a text, you rewrite it for a 4th grader, keeping the folowing in mind:

- The original structure of the text should be preserved 1:1
- You make sure that the information content is exactly the same. You do not add or remove any information (you are allowed to add short clarifying passages to explain words or concepts that might be unclear to a 4th grader)
- You also explain acronyms if they are not explained in the text itself
- You mostly just simplify vocabulary and phrase structures.

Make sure you don't inadvertently alter the meaning of the original text.

You also fix the text formatting by removing references, fixing spacing issues, and replacing HTML tags with markdown equivalents when possible. 
"""

    messages_for_openai = [
        OpenaiChatMessage(role="system", content=prompt),
        OpenaiChatMessage(
            role="user",
            content=f"""   
    {text}
    """,
        ),
    ]

    formatted_text = await get_response_openai_nonstream(
        messages_for_openai,
    )

    print(f"Original text: \n\n {text} \n\n Formatted text: \n\n {formatted_text}")
    return formatted_text

if __name__ == "__main__":
    import asyncio

    async def main():
        async with aiohttp.ClientSession() as session:
            pages = await fetch_relevant_wikipedia_pages("Baseball", session)
            sections = []
            for page in pages:
                sections += extract_sections(page)
            
            print(f"Got {len(sections)} sections")
            # Rank all sections in parallel 
            start_time = time.time()
            results = await asyncio.gather(*[rank_section(section[0], section[1], "Baseball") for section in sections[:10]])
            print(f"Total time: {time.time() - start_time}")

            with open("results.json", "w") as f:
                json.dump(results, f, indent=4)

            best_text = select_best_text(results)
            print(f"Best text: \n\n {best_text}")
        
    asyncio.run(main())