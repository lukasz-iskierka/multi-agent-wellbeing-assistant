from src.schemas.models import SearchQuery
from src.schemas.states import ConsultationState, ConsultationOutputState
from src.utils.logging_utils import log

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from langgraph.types import interrupt, Send, Command
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.messages import get_buffer_string, RemoveMessage
from typing import List, Sequence
from langchain_tavily import TavilySearch
from langchain_community.document_loaders import WikipediaLoader





def build_consultation_subgraph():

    # Instatiate chat models
    llm_4o = ChatOpenAI(model="gpt-4o-2024-11-20", temperature=0) 
    llm_4_1_mini = ChatOpenAI(model="gpt-4.1-mini-2025-04-14", temperature=0)
  
    # Nodes and edges

    question_instructions = """# Identity and objectives:
    You are a client who is having an appointment with a wellbeing practitioner. Your objective is to receive in-depth advice tailored to your problem:

    {problem}

    You finished your previous appointment with the following helpful advice form the practitioner:

    {advice}

    # Follow these steps:
    1. Review your problem.
    2. Review the piece of advice you have previously received.
    2. Review your current conversation with the practitioner.
    3. Review the (optional) summary of the conversation with the practitioner:

    {summary}

    3. Create a persona that fits the problem you came to discuss and stay in your character throughout the consultation.
    3. Begin by greeting the practitioner and ask a follow up question regarding the piece of advice you have received during the last appointment.
    4. Continue asking questions until you think the practitioner has offered enough help and clarification on the piece of advice.
    5. Be curious, critical about practicalities of the advice you received and ask in-depth questions. Ask for real-world implementations.
    6. IMPORTANT: When you feel you don't need more information and all your questions have been answered, finish the consultation by stating: "Thank you and goodbye!"   
    """

    def question_generator(state: ConsultationState):
        
        """Node to genarate a question for a single step in the wellbeing action plan."""

        problem = state["problem"]
        step = state["step"]
        conversation = state.get("messages", [])
        summary = state.get("summary", "")

        formatted_question_instructions = question_instructions.format(
            problem = problem,
            advice = step.step_summary,
            summary=summary
        )

        messages = [
            SystemMessage(content=formatted_question_instructions),
            AIMessage(content="Hello! What brings you here today?", name="practitioner") # Prompt the simulated conversation
        ]

        question = llm_4o.invoke(messages + conversation)
        question.name = "client"

        return {"messages": [question]}


    def skip_the_search(state: ConsultationState) -> Sequence[str]:
        
        """Logic to determine if the simulated consultation was concluded so that the web/wiki search can be skipped."""

        latest_message = state["messages"][-1]

        if "Thank you and goodbye!" in latest_message.content:
            # Jump to answer_generation node
            return "answer_generator"
        else:
            # Continue with the parallel web and wiki search
            return ["web_query_constructor", "wiki_query_constructor"]


    web_query_instructions = """# Identity and objectives:
    You are an assistant specialised in creating quality and well-structured search queries for use in web-search. 
    You will be given a conversation between a client and a wellbeing practitioner and your goal is to create a query based on that conversation. 

    # Follow these steps:
    1. Analyse the problem the client came to discuss with the practitioner:

    {problem}

    2. Analyse the conversation.
    3. Analyse (optional) summary of the previous parts of the conversation:

    {summary}

    4. IMPORTANT: Pay particular attention to the final question posed by the client.
    5. Convert this final question into a well-structured web search query"""

    def web_query_constructor(state: ConsultationState):
        
        """Node to construct a web search query according to the expected output schema."""

        problem = state["problem"]
        conversation = state["messages"]
        summary = state.get("summary", "")

        formatted_query_instructions = web_query_instructions.format(
            problem=problem,
            summary=summary
        )

        # Force output format
        structured_llm = llm_4o.with_structured_output(SearchQuery)
        # Generate the query
        query = structured_llm.invoke([formatted_query_instructions] + conversation)

        return {"webquery": query.search_query}


    wiki_query_instructions = """# Identity and objectives:
    You are an assistant specialised in creating quality and well-structured search queries for use in Wikipedia search. 
    You will be given a conversation between a client and a wellbeing practitioner and your goal is to create a query based on that conversation. 

    # Follow these steps:
    1. Analyse the problem the client came to discuss with the practitioner:

    {problem}

    2. Analyse the conversation.
    3. Analyse (optional) summary of the previous parts of the conversation:

    {summary}

    4. IMPORTANT: Pay particular attention to the final question posed by the client.
    5. Convert this final question into a well-structured Wikipedia search query
    6. When constructing the query, use these pointers:
    * Use specific, unique terms - Search "Fermi paradox" instead of "aliens exist"
    * Include proper names when known - Search "Marie Curie radium" instead of "female scientist radioactivity"
    * Add disambiguating context for common terms - Search "Python programming" instead of just "Python"
    * Use the most common name or spelling - Search "World War II" instead of "Second World War" or "WW2"
    * Combine key concepts with AND - Search "Einstein AND photoelectric" instead of "Einstein's work on light"
    * Keep the queries short.
    """
    def wiki_query_constructor(state: ConsultationState):
        
        """Node to construct a Wikipedia search query according to the expected output schema."""

        problem = state["problem"]
        conversation = state["messages"]
        summary = state.get("summary", "")

        formatted_query_instructions = wiki_query_instructions.format(
            problem=problem,
            summary=summary
        )

        structured_llm = llm_4o.with_structured_output(SearchQuery)
        query = structured_llm.invoke([formatted_query_instructions] + conversation)

        return {"wikiquery": query.search_query}


    def websearch(state: ConsultationState):
        
        "Node to perform the websearch with constructed query and to save the source docs"
        
        webquery = state["webquery"]

        tavily = TavilySearch(
            max_results=2,
            topic="general",
            include_raw_content=True # For more data
            )
        
        # Run the web search and return docs
        docs = tavily.invoke(input=webquery)

        def raw_content_snippet(doc, max_length=1500):
            
            """Limit the length of the scraped raw content."""
            
            raw_content = doc.get("raw_content", "")

            if raw_content:
                if len(raw_content) > max_length:
                    return raw_content[:max_length]
                else:
                    return raw_content
            else:
                return ""

        # Format all returned docs
        formatted_docs = "\n\n-----\n\n".join(
            [
                f'<Document source: {doc["url"]}, title: "{doc["title"]}"/>\n\n{doc.get("content", "")}\n\n{raw_content_snippet(doc)}\n</Document>'
                for doc in docs['results']
            ]
        )

        return {"source_docs": [formatted_docs]}


    def wikisearch(state: ConsultationState):
        
        "Node to perform Wikipedia search with constructed query and to save the source docs"

        wikiquery = state["wikiquery"]

        # Run the wiki search and return found docs
        docs = WikipediaLoader(
            query=wikiquery, 
            load_max_docs=2, 
            doc_content_chars_max=1500
            ).load() 
        
        # Format all returned docs
        formatted_docs = "\n\n-----\n\n".join(
            [
                f'<Document source: {doc.metadata["source"]}, title: "{doc.metadata["title"]}"/>\n{doc.page_content}\n</Document>'
                for doc in docs
            ]
        )

        return {"source_docs": [formatted_docs]}


    answer_instructions = """# Identity and objectives:
    You are an expert wellbeing practitioner who is having an appointment with a client. Your goal is to answer all questions coming from your client, while taking into account:

    - The client's problem they came to discuss with you:

    {problem}

    - The context (knowledge) that is available to you:

    {context}

    - The conversation you're having with the client.

    # When answering the client's questions follow these steps:
    1. Review the client's problem.
    2. Review the context (knowledge) that is available to you. Do not introduce external information or make assumptions beyond what is explicitly stated in the context.
    3. Review your current conversation.
    4. Review the (optional) summary of the earlier parts of the appointment:

    {summary}

    5. Begin by welcoming the client and move on to answering their questions based on the context (knowledge) that you have. Only use the information provided in the context. 
    6. The context contains sources at the topic of each individual document.
    7. Include these sources to your answer next to any relevant statements. For example, for source # 1 use [1]. 
    8. List your sources in order at the bottom of your answer. [1] Source # 1, [2] Source # 2, etc
    9. If the source is: <Document source: https://positivepsychology.com/mindfulness-based-stress-reduction-mbsr/>' then just list: 
            
    [1] https://positivepsychology.com/mindfulness-based-stress-reduction-mbsr

    10. Make sure to include the source the whole domain, so don't skip the 'https://'        
    10. Skip the addition of the brackets as well as the Document source preamble in your citation.""" 

    def answer_generator(state: ConsultationState):
    
        "Node to generate the practitioner's answer based on the source docs."
        
        problem = state["problem"]
        context = "\n\n-----\n\n".join([doc for doc in state["source_docs"][-2:]]) # Only include the last two docs (Web + Wiki)
        summary = state.get("summary", "")
        conversation = state["messages"]
        latest_message = conversation[-1]
        cycles_counter = state.get("cycles_counter", 0)

        # If consultation was concluded, add a mock goodbye message from the practitioner and don't invoke the LLM
        if "Thank you and goodbye!" in latest_message.content:
            cycles_counter += 1
            return {
                "messages": [AIMessage(content="If you have any more questions in the future or need further support, don't hesitate to reach out. Take care and goodbye!", name="practitioner")],
                "cycles_counter": cycles_counter
                }
        # Otherwise, format the answer with web/wiki docs and invoke the LLM to generate the answer
        else:
            formatted_answer_instructions = answer_instructions.format(
                problem=problem,
                context=context,
                summary=summary
            )

            sys_message = [SystemMessage(content=formatted_answer_instructions)]

            answer = llm_4o.invoke(sys_message + conversation)
            answer.name = "practitioner"

            cycles_counter += 1

            return {
                "messages": [answer],
                "cycles_counter": cycles_counter
                }


    def save_the_transcript(state: ConsultationState):
        
        """Node to save the consultation transcript."""
        
        conversation = state["messages"]
        transcript = state.get("transcript", "")

        # Include the last round of conversation between the client and the practitioner
        if transcript:
            new_entries = get_buffer_string(conversation[-2:]) 
        # If the transcript is still empty
        else:
            new_entries = get_buffer_string([AIMessage(content="Hello! What brings you here today?", name="practitioner")] + conversation)

        transcript += f"\n{new_entries}"

        return {"transcript": transcript}


    def continue_consultation(state: ConsultationState):
        
        """Conditional edge to decide if the consultation should continue or if it should end."""
        
        conversation = state["messages"]
        max_cycles = state.get("max_cycles", 2)
        cycles_counter = state["cycles_counter"]

        # If consultation was concluded proceed to the write-up stage
        if cycles_counter >= max_cycles or "Thank you and goodbye!" in conversation[-2].content:     
            return "section_writer"
        # Otherwise, continue the consultation (with summary generation step)
        else:
            return "generate_summary"
        

    summary_instructions = """# Identity and ojectives:
    Your are an assistant specialised at summarising conversations.

    # Follow these steps:
    1. Review the history of conversation.
    2. Review the (optional) previous summary:

    {summary}

    3. Summarise the history of conversation making sure to preserve all important details, including who said what.
    4. IMPORTANT: If previous summary was supplied, extend it with the new one.
    5. Do not exceed 200 words.
    """

    def generate_summary(state: ConsultationState):
        
        """Node to generate a summary of the consultation if it runs too long."""

        conversation = state["messages"]
        summary = state.get("summary", "")
        
        summary_instructions_formatted = summary_instructions.format(summary=summary)

        # Summarise the consultation to save on tokens
        if len(conversation) >= 6:
            summary = llm_4_1_mini.invoke([summary_instructions_formatted] + conversation)
            # Only keep the last round of conversation between the client and the practitioner
            messages_to_remove = [RemoveMessage(id=message.id) for message in conversation[:4]]
            return {"summary": summary.content, "messages": messages_to_remove}
        else:
            pass

    section_writer_instructions = """# Identity and objectives:
    You are an expert technical writer. 
    Your task is to create a short and actionable section of a Wellbeing Action Plan focused on a specific step from the plan while considering the problem reported by a client. 
    This section should only be based on a transcript form a consultation between a client and a wellbeing practitioner and initial helpful tip the client received. Do not use external resources.
    The transcript includes a few questions the client asked during the appointment and answers from the practitioner with some helpful advice.
    Each piece of advice from the practitioner is accompanied by the in-text source indicated by square brackets, e.g. [1] with full list of sources at the bottom of the answer, e.g. [1] https://positivepsychology.com/cbt-therapy

    1. Analyse the content of the transcript: 

    Transcript:

    {transcript}

    ---

    2. Analyse the specific step from the plan (pay attention to the helpful tip):

    Step:

    {step}

    ---

    3. Create the section structure using markdown formatting:
    - Use ## for the section title
    - Use ### for sub-section headers
            
    3. Write the section of the plan following this structure:
    a. Title (## header)
    b. Summary (### header)
    c. Sources (### header)

    4. Make your section title engaging based upon the step from the Wellbeing Action Plan.

    ---

    5. For the summary section:
    - Begin the summary with general background / context related to the step from the Wellbeing Action Plan tied to the problem reported by the client.
    - Pay special attention to the pieces of advice for which the client expressed the biggest enthusiasm / asked questions.
    - However, do not mention actual concerns or emotions expressed by the client that were captured in the transcript. Treat the transcript as a soource of information only.
    - Use easily digestable language.
    - Do not mention names of the client or the practitioner (if present in the transcript).
    - Aim for 300-400 words.
    - Create a numbered list of source documents, as you use them in the summary.
    - Use numbered sources in your report (e.g., [1], [2]) based on information from source documents
            
    6. In the Sources section:
    - Include all sources used in your summary assigning them a unique number
    - Provide full links to relevant websites or specific document paths
    - Separate each source by a newline. Use two spaces at the end of each line to create a newline in Markdown
    - It will look like:

    ### Sources
    [1] Link or Document name
    [2] Link or Document name

    7. Be sure to combine sources. For example this is not correct:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
    [4] https://ai.meta.com/blog/meta-llama-3-1/

    There should be no redundant sources. It should simply be:

    [3] https://ai.meta.com/blog/meta-llama-3-1/
            
    8. Final review:
    - Ensure the completed section follows the required structure.
    - Make sure there is only one sources section an that numbered sources have been used throughout the summary 
    - Include no preamble before the title of the Wellbeing Action Plan
    - Check that all guidelines have been followed"""

    def section_writer(state: ConsultationState):
        
        """Node to write an actionable entry for the wellbeing action plan based on the consultation transcript."""

        step = state["step"]
        theme = step.theme
        transcript = state["transcript"]
        problem = state["problem"]
    
        formatted_writing_instructions = section_writer_instructions.format(
            step=step.step_summary,
            transcript=transcript
        )

        messages = [
            SystemMessage(content=formatted_writing_instructions),
            HumanMessage(content=f"Write a section for my Wellbeing Action Plan, in the context of my problem: {problem}")
        ]
        section = llm_4o.invoke(messages)

        # print progress log
        if section.content:
            log(f"[Consultation] Section for theme '{theme}' successfully generated!")

        return {"sections": [section.content]}


    # build the subgraph

    # Add nodes
    builder = StateGraph(state_schema=ConsultationState, output_schema=ConsultationOutputState)
    builder.add_node(question_generator)
    builder.add_node(web_query_constructor)
    builder.add_node(wiki_query_constructor)
    builder.add_node(websearch)
    builder.add_node(wikisearch)
    builder.add_node(answer_generator)
    builder.add_node(save_the_transcript)
    builder.add_node(generate_summary)
    builder.add_node(section_writer)

    # Add edges (logic)
    builder.add_edge(START, "question_generator")
    builder.add_conditional_edges("question_generator", skip_the_search, ["answer_generator", "wiki_query_constructor", "web_query_constructor"])
    builder.add_edge("wiki_query_constructor", "wikisearch")
    builder.add_edge("web_query_constructor", "websearch")
    builder.add_edge(["websearch", "wikisearch"], "answer_generator")
    builder.add_edge("answer_generator", "save_the_transcript")
    builder.add_conditional_edges("save_the_transcript", continue_consultation, ["generate_summary", "section_writer"])
    builder.add_edge("generate_summary", "question_generator")

    # Compile the subgraph and return
    return builder.compile()