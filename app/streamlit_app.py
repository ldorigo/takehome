

import json 
import asyncio
from time import time
import aiohttp
import pandas as pd
import streamlit as st
from app.frq import generate_frqs, select_best_frq, assess_frq
from wikitext import clean_and_format_text, extract_sections, fetch_relevant_wikipedia_pages, rank_section, select_best_text
from feedback import give_feedback_on_answer, rewrite_text_according_to_feedback
from student import answer_question_as_student

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
    results = await asyncio.gather(*[rank_section(section[0], section[1],topic) for section in sections])
    print(f"Total time: {time() - start_time}")
    
    print(f"Got text rankings, selecting the best one..."   )
    best_text = select_best_text(results)

    best_text_formatted = await clean_and_format_text(best_text["text"])
    return best_text_formatted, best_text


@st.cache_data
def get_best_text_sync(sections, topic):
    loop = asyncio.get_event_loop()
    best_text_formatted, best_text = loop.run_until_complete(get_best_text(sections, topic))
    return best_text_formatted, best_text

@st.cache_data
def generate_frqs_sync(text):
    loop = asyncio.get_event_loop()
    frqs = loop.run_until_complete(generate_frqs(text))
    return frqs

@st.cache_data
def rank_frqs_sync(frqs, text):
    loop = asyncio.get_event_loop()
    frq_rankings = loop.run_until_complete(asyncio.gather(*[assess_frq(frq, text) for frq in frqs]))
    return frq_rankings





def main():
    print("Rendering app")
    st.title("AI Writing Mentor")
    st.write(f"Hi! I'm here to help you practice your writing skills on free-response questions. Could you choose a topic that interests you below?")

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
            st.error(f"Sorry, I couldn't find any relevant texts for {topic}. Please try another topic.")
            return

        with st.spinner(f"I found {len(sections)} potential texts - selecting the best one for you..."):
            best_text_formatted, best_text =  get_best_text_sync(sections, topic)

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
                display_names = ["Relevance", "Age Appropriateness", "Complexity Fit", "Potential for CCSS assessment", "Overall Educational Value"]
                key_names = ['relevance', 'age_appropriateness', 'complexity_fit', 'potential_for_assessment', 'overall_educational_value']

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
                    """)
                st.table(
                    to_display_df
                )
                st.markdown(
                    f"Overall score: {overall_score}"
                )
        # with st.form("question_form"):
        with st.status("I'm finding a good question for you, hang on tight!"):
            
            st.write(f"Generating a few candidate questions...")
            frqs = generate_frqs_sync(best_text_formatted)

            st.write(f"I generated {len(frqs)} questions for you. Let me select the best one...")

            # Rank them in parallel
            start_time = time()
            frq_rankings = rank_frqs_sync(frqs, best_text_formatted)
            # frq_rankings = await asyncio.gather(*[assess_frq(frq, text) for frq in frqs])
            print(f"Assessment time: {time() - start_time}")

            best_frq = select_best_frq(frq_rankings)

        st.markdown(f"## {best_frq['frq']}")

        st.write(f"Here's the question I selected for you. Now, write your answer in the box below. You should try to write at least 200 words. Remember to use the text above to help you answer the question.")

        with st.expander("*Why was this question chosen?*", expanded=False):
            st.write(f"*Note: this would not be shown to the user, it's to help understand the prototype*")
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

            st.table(
                to_display_df
            )

            st.markdown(            
                f"Overall score: {overall_score} (this scores is a weighted average of the scores above)"
            )
            
        generated_answer = st.session_state.get("generated_answer", "")
        generating_answer = st.session_state.get("generating_answer", False)
        answer = st.text_area("Your answer",generated_answer, height=300, disabled=generating_answer)



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
                    response_prompt="A terrible answer. The answer does not answer the question, and is completely unstructured and full of non-sequiturs. It's also FULL of grammatical errors and typos, badly formatted, and hard to read."
                print(f"Gnerating answer with selected: {selected}")
                generated_answer = loop.run_until_complete(answer_question_as_student(best_frq["frq"], best_text_formatted, response_prompt))

            st.session_state["generated_answer"] = generated_answer
            st.session_state["generating_answer"] = False
            st.session_state["selected"] = "Choose answer quality"

            st.experimental_rerun() 

        # # st.button("Generate an answer", key="generate_answer", disabled=generating_answer, on_click=lambda: st.session_state.update({"generating_answer": True}))
        # # Replace with a select dropdown that allows choosing a good, mediocre or bad answer.
        # selected_idx = st.session_state.get("selected_idx", 0)
        # print(f"Selected idx: {selected_idx}")

        # selected = st.selectbox("Generate an answer (choose quality)", options = ["Choose answer quality", "Good", "Mediocre", "Bad"],index=selected_idx,  disabled=generating_answer)


        # print(f"Selected: {selected}")

        # if selected != "Choose answer quality":
        #     print("Selected and not generating answer")
        #     st.session_state["selected"] = selected
        #     st.session_state["selected_idx"] = ["Choose answer quality", "Good", "Mediocre", "Bad"].index(selected)
        #     st.session_state["generating_answer"] = True
        #     st.experimental_rerun()

        # This is not working - replace with 3 buttons in 3 columns, "Generate good answer", "Generate mediocre answer", "Generate bad answer"

        left, mid, right = st.columns(3)
        with left:
            good_answer = st.button("Generate a good answer", key="generate_good_answer", disabled=generating_answer, on_click=lambda: st.session_state.update({"selected": "good", "generating_answer": True}))
        with mid:
            mediocre_answer = st.button("Generate a mediocre answer", key="generate_mediocre_answer", disabled=generating_answer, on_click=lambda: st.session_state.update({"selected": "mediocre", "generating_answer": True}))
        with right:
            bad_answer = st.button("Generate a bad answer", key="generate_bad_answer", disabled=generating_answer, on_click=lambda: st.session_state.update({"selected": "bad", "generating_answer": True}))

        def set_answer(answer):
            print(f"Pressed submit answer, answer: {answer}")
            st.session_state["answer"] = answer
            st.session_state["generating_answer"] = False
        submit_answer = st.button("Submit answer", key="submit_answer", disabled=generating_answer, on_click=lambda: set_answer(answer))
            

        if submit_answer:
            print("Submitting answer")
            answer = st.session_state["answer"]

            st.divider()


            st.markdown("## Feedback on your answer")

            with st.spinner("I'm evaluating your answer and generating some feedback for you. Hang on tight!"):


                feedbacks = loop.run_until_complete(give_feedback_on_answer(answer, best_frq["frq"], best_text_formatted))
                print("Done generating feedback")

            for feedback_category, feedback in feedbacks.items():
                grade = f"{feedback['aggregated_grade']}/5"
                color = "green" if feedback['aggregated_grade'] > 3 else "black" if feedback['aggregated_grade'] == 3 else "red"
                label = f"""## {feedback_category}: :{color}[{grade}]
{feedback['aggregated_summary']}
*(click here to view detailed feedback)*
    """        

                with st.expander(label=label, expanded=False):
                    st.write(f"""
#### Detailed feedback: 
{feedback['aggregated_feedback']}
""")
                

            with st.spinner("Writing a new answer that incorporates the feedback..."):
                    rewritten_answer= loop.run_until_complete(rewrite_text_according_to_feedback(text=best_text_formatted, question=best_frq["frq"], answer=answer, feedback=feedbacks))
                             
            with st.expander("View how you could have rewritten your answer to improve it", expanded=False):
                st.write(f"""
{rewritten_answer}
""")

            


if __name__ == '__main__':
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