import asyncio
from time import time
import aiohttp
import pandas as pd
import streamlit as st
from frq import generate_frqs, select_best_frq, assess_frq
from wikitext import (
    clean_and_format_text,
    extract_sections,
    fetch_relevant_wikipedia_pages,
    rank_section,
    select_best_text,
)
from feedback import give_feedback_on_answer, rewrite_text_according_to_feedback
from student import answer_question_as_student

# from llm import openai
import llm


async def get_sections(topic):
    async with aiohttp.ClientSession() as session:
        print("Fetching relevant wikipedia pages")
        pages = await fetch_relevant_wikipedia_pages(topic, session)
        print("Fetched relevant wikipedia pages, extracting sections")

        sections = []
        for page in pages:
            sections += extract_sections(page)

        print(f"Found {len(sections)} sections")
        return sections


@st.cache_data
def get_sections_sync(topic):
    loop = asyncio.get_event_loop()
    sections = loop.run_until_complete(get_sections(topic))
    return sections


async def get_best_text(sections, topic):
    start_time = time()
    results = await asyncio.gather(
        *[rank_section(section[0], section[1], topic) for section in sections]
    )
    print(f"Total time: {time() - start_time}")

    print(f"Got text rankings, selecting the best one...")
    best_text = select_best_text(results)

    best_text_formatted = await clean_and_format_text(best_text["text"])
    return best_text_formatted, best_text


@st.cache_data
def get_best_text_sync(sections, topic):
    loop = asyncio.get_event_loop()
    best_text_formatted, best_text = loop.run_until_complete(
        get_best_text(sections, topic)
    )
    return best_text_formatted, best_text


@st.cache_data
def generate_frqs_sync(text):
    loop = asyncio.get_event_loop()
    frqs = loop.run_until_complete(generate_frqs(text))
    return frqs


@st.cache_data
def rank_frqs_sync(frqs, text):
    loop = asyncio.get_event_loop()
    frq_rankings = loop.run_until_complete(
        asyncio.gather(*[assess_frq(frq, text) for frq in frqs])
    )
    return frq_rankings


@st.cache_data
def answer_question_sync(best_frq, best_text_formatted, response_prompt):
    loop = asyncio.get_event_loop()
    answer = loop.run_until_complete(
        answer_question_as_student(
            best_frq["frq"], best_text_formatted, response_prompt
        )
    )
    return answer


@st.cache_data
def give_feedback_sync(answer, frq, text):
    loop = asyncio.get_event_loop()
    feedbacks = loop.run_until_complete(give_feedback_on_answer(answer, frq, text))
    return feedbacks


@st.cache_data
def rewrite_text_according_to_feedback_sync(text, question, answer, feedback):
    loop = asyncio.get_event_loop()
    rewritten_answer = loop.run_until_complete(
        rewrite_text_according_to_feedback(
            text=text, question=question, answer=answer, feedback=feedback
        )
    )
    return rewritten_answer


def main():
    print("Rendering app")
    st.title("AI Writing Mentor")
    #
    with st.sidebar:
        openai_key_input = st.text_input("Openai Key", key="openai_key")
        model_toggle = st.radio(
            "Model", options=["GPT-3.5", "GPT-4"], index=0, key="model_toggle"
        )
        if model_toggle == "GPT-4":
            st.write(
                f"*WARNING: GPT-4 gives much better results but is slow (~40 sec/step) and expensive (~1-2$ for a full run)*"
            )

        clear = st.button("Clear cache")
        if clear:
            st.cache_data.clear()
            st.cache_resource.clear()
            # also clear the session state
            st.session_state.clear()
            st.experimental_rerun()

    if not openai_key_input:
        st.error("Please enter an OpenAI key in the sidebar to continue.")
        return

    llm.openai.api_key = openai_key_input
    llm.openai.MODEL = "gpt-4-0613" if model_toggle == "GPT-4" else "gpt-3.5-turbo-0613"
    print(f"Set model to {llm.MODEL} (toggle: {model_toggle})")

    st.write(
        f"Hi! I'm here to help you practice your writing skills on free-response questions. Could you choose a topic that interests you below?"
    )

    with st.form("topic_form"):
        topic = st.text_input("Topic of interest", "Baseball")
        submit_topic = st.form_submit_button("Give me a question!")

    if submit_topic or st.session_state.get("topic"):
        # save the topic in the session state so we don't have to re-enter it
        st.session_state["topic"] = topic
        # When the button is clicked, fetch the relevant wikipedia pages - indicate that in a status text

        with st.spinner("I'm looking for relevant texts..."):
            sections = get_sections_sync(topic)
        if not sections:
            st.error(
                f"Sorry, I couldn't find any relevant texts for {topic}. Please try another topic."
            )
            return

        with st.spinner(
            f"I found {len(sections)} potential texts - selecting the best one for you..."
        ):
            best_text_formatted, best_text = get_best_text_sync(sections, topic)

        with st.container():
            # st.header("This is the text. Read it carefully, and then answer the question below.")
            # st.subheader(f"{best_text['title']}")
            st.markdown(
                f"""## Here's the text I selected for you.

*Read it carefully, and then answer the question below.*

### {best_text["title"]}

{best_text_formatted}
"""
            )

            with st.expander("*Why was this text chosen?*", expanded=False):
                to_display_data = []
                display_names = [
                    "Relevance",
                    "Age Appropriateness",
                    "Complexity Fit",
                    "Potential for CCSS assessment",
                    "Overall Educational Value",
                ]
                key_names = [
                    "relevance",
                    "age_appropriateness",
                    "complexity_fit",
                    "potential_for_assessment",
                    "overall_educational_value",
                ]

                for key_name, display_name in zip(key_names, display_names):
                    to_display_data.append(
                        {
                            "Criterium": display_name,
                            "Score (1-5)": best_text[key_name + "_score"],
                            "Reasoning": best_text[key_name + "_reasoning"],
                        }
                    )
                to_display_df = pd.DataFrame(to_display_data)
                # use criterium as index
                to_display_df.set_index("Criterium", inplace=True)
                overall_score = to_display_df["Score (1-5)"].mean()

                st.markdown(
                    f"""
                    *Note: this would not be shown to the user, it's to help understand the prototype*
                    """
                )
                st.table(to_display_df)
                st.markdown(
                    f"Overall score: {overall_score:.2f} (this scores is a weighted average of the scores above)"
                )

                st.divider()

                st.markdown(f"### This was the original text before simplification: ")
                # Write the text, escape any possible markdown:
                st.write(
                    best_text["text"]
                    .replace("*", "\*")
                    .replace("_", "\_")
                    .replace("#", "\#")
                    .replace("+", "\+")
                    .replace("=", "\=")
                    .replace("|", "\|")
                    .replace("{", "\{")
                    .replace("}", "\}")
                    .replace("!", "\!")
                    .replace("[", "\[")
                    .replace("]", "\]")
                    .replace("`", "\`")
                    .replace("~", "\~")
                    .replace(">", "\>")
                    .replace("<", "\<")
                    .replace("$", "\$")
                    .replace(":", "\:")
                    .replace("?", "\?")
                    .replace("/", "\/")
                    .replace("\\", "\\\\")
                )
        # with st.form("question_form"):
        with st.status("I'm finding a good question for you, hang on tight!"):
            st.write(f"Generating a few candidate questions...")
            frqs = generate_frqs_sync(best_text_formatted)

            st.write(
                f"I generated {len(frqs)} questions for you. Let me select the best one..."
            )

            # Rank them in parallel
            start_time = time()
            frq_rankings = rank_frqs_sync(frqs, best_text_formatted)
            # frq_rankings = await asyncio.gather(*[assess_frq(frq, text) for frq in frqs])
            print(f"Assessment time: {time() - start_time}")

            best_frq = select_best_frq(frq_rankings)

        st.markdown(f"## {best_frq['frq']}")

        st.write(
            f"Here's the question I selected for you. Now, write your answer in the box below. You should try to write at least 100 to 200 words. Remember to use the text above to help you answer the question!"
        )

        with st.expander("*Why was this question chosen?*", expanded=False):
            st.write(
                f"*Note: this would not be shown to the user, it's to help understand the prototype*"
            )
            st.write(f"The question was ranked best using the following criteria:")

            to_display_data = []

            display_names = [
                "Clarity",
                "Alignment with Standard",
                "Age Appropriateness",
                "Analytical Depth",
                "Open-Endedness",
                "Textual Scope",
                "Language Complexity",
                "Bias-Free",
                "Use of Action Verbs",
                "Feasibility of Answer",
            ]

            key_names = [
                "clarity",
                "alignment",
                "age_appropriateness",
                "analytical_depth",
                "open_endedness",
                "textual_scope",
                "language_complexity",
                "bias_free",
                "action_verbs",
                "feasibility_of_answer",
            ]

            for key_name, display_name in zip(key_names, display_names):
                to_display_data.append(
                    {
                        "Criterium": display_name,
                        "Score (1-5)": best_frq[key_name + "_score"],
                        "Reasoning": best_frq[key_name + "_reasoning"],
                    }
                )

            to_display_df = pd.DataFrame(to_display_data)

            to_display_df.set_index("Criterium", inplace=True)
            overall_score = best_frq["score"]

            st.table(to_display_df)

            st.markdown(
                f"Overall score: {overall_score:.2f} (this scores is a weighted average of the scores above)"
            )

        generated_answer = st.session_state.get("generated_answer", "")
        generating_answer = st.session_state.get("generating_answer", False)
        answer = st.text_area(
            "Your answer", generated_answer, height=300, disabled=generating_answer
        )

        if generating_answer:
            print("Generating answer")
            with st.spinner("Generating an answer..."):
                selected = st.session_state["selected"]
                response_prompt = ""
                if selected == "good":
                    response_prompt = "Excellent answer, the best that can possibly be expected from a fourth-grader. All of the relevant information is included, and the answer is well-structured and easy to follow. The answer is also free of grammatical errors and typos."
                elif selected == "mediocre":
                    response_prompt = "A mediocre answer. The answer is somewhat relevant, but it is not well-structured and it is hard to follow. There are a few grammatical errors and typos, but the answer is still readable."
                elif selected == "bad":
                    response_prompt = "A terrible answer. The answer does not answer the question, and is completely unstructured and full of non-sequiturs. It's also FULL of grammatical errors and typos, badly formatted, and hard to read."
                print(f"Generating answer with selected: {selected}")
                generated_answer = answer_question_sync(
                    best_frq, best_text_formatted, response_prompt
                )

            st.session_state["generated_answer"] = generated_answer
            st.session_state["generating_answer"] = False
            st.session_state["selected"] = "Choose answer quality"

            st.experimental_rerun()
        left, mid, right = st.columns([1, 1, 1])

        with left:
            good_answer = st.button(
                "Generate a good answer",
                key="generate_good_answer",
                disabled=generating_answer,
                on_click=lambda: st.session_state.update(
                    {"selected": "good", "generating_answer": True}
                ),
            )
        with mid:
            mediocre_answer = st.button(
                "Generate a mediocre answer",
                key="generate_mediocre_answer",
                disabled=generating_answer,
                on_click=lambda: st.session_state.update(
                    {"selected": "mediocre", "generating_answer": True}
                ),
            )
        with right:
            bad_answer = st.button(
                "Generate a bad answer",
                key="generate_bad_answer",
                disabled=generating_answer,
                on_click=lambda: st.session_state.update(
                    {"selected": "bad", "generating_answer": True}
                ),
            )

        def set_answer(answer):
            print(f"Pressed submit answer, answer: {answer}")
            st.session_state["answer"] = answer
            st.session_state["generating_answer"] = False

        submit_answer = st.button(
            "Submit answer",
            key="submit_answer",
            disabled=generating_answer,
            on_click=lambda: set_answer(answer),
        )

        if submit_answer:
            print("Submitting answer")
            answer = st.session_state["answer"]

            st.divider()

            st.markdown("## Feedback on your answer")

            with st.spinner(
                "I'm evaluating your answer and generating some feedback for you. Hang on tight!"
            ):
                # feedbacks = loop.run_until_complete(give_feedback_on_answer(answer, best_frq["frq"], best_text_formatted))
                feedbacks = give_feedback_sync(answer, best_frq, best_text_formatted)
                print("Done generating feedback")

            for feedback_category, feedback in feedbacks.items():
                grade = f"{feedback['aggregated_grade']}/5"

                color = (
                    "green"
                    if feedback["aggregated_grade"] > 3
                    else "black"
                    if feedback["aggregated_grade"] == 3
                    else "red"
                )
                label = f"""## {feedback_category}: :{color}[{grade}]
{feedback['aggregated_summary']}
*(click here to view detailed feedback)*
    """

                with st.expander(label=label, expanded=False):
                    newline = "\n"
                    st.write(
                        f"""
#### Detailed feedback: 
{feedback['aggregated_feedback'].replace(newline, '  ' + newline)}
"""
                    )

            with st.spinner("Writing a new answer that incorporates the feedback..."):
                # rewritten_answer= loop.run_until_complete(rewrite_text_according_to_feedback(text=best_text_formatted, question=best_frq["frq"], answer=answer, feedback=feedbacks))
                rewritten_answer = rewrite_text_according_to_feedback_sync(
                    text=best_text_formatted,
                    question=best_frq["frq"],
                    answer=answer,
                    feedback=feedbacks,
                )

            st.write("### Good job completing this exercise! \n If you'd like click on the box below to view an example of how you could rewrite your answer to incorporate this feedback. Otherwise, feel free to try answering again to see how you do - or choose a new topic to start over!")
            with st.expander(
                "View how you could have rewritten your answer to improve it",
                expanded=False,
            ):
                st.write(
                    f"""
{rewritten_answer}
"""
                )


if __name__ == "__main__":
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        main()
        # asyncio.run(main())
    except KeyboardInterrupt:
        pass
    finally:
        print("Closing Loop")
        loop.close()
