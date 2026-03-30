from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
# from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from typing import TypedDict, Dict, List, Optional, Literal, Annotated, Any
from own_framework_prompts import *
from pydantic import BaseModel, Field
from dotenv import load_dotenv
load_dotenv()

class AgentOutputStage1(BaseModel):
    probability: float = Field(..., description="The probability that the error is present", ge=0.0, le=1.0)
    reason: str = Field(..., description="Explanation pointing to concrete words, phrases, or structures that justify the probability.")
    confidence: float = Field(..., description="Score out of 100 on how confident you are that this error is present in the sentence.", ge=0.0, le=100.0)

class AgentOutputStage2(BaseModel):
    reEvaluatedProb: float = Field(..., description="Based on your evaluation re evaluate the probability while considering the probability given by the previous agent.", ge=0.0, le=1.0)
    thoughtsOnStage1: str = Field(..., description="Based on your evaluations, what are your thoughts on evaluations of the previous agent?")
    reason: str = Field(..., description="Give a brief explanation of your thoughts on the previous agent's evaluations. If you agree of disagree with the previous evaluation give concrete evidence for your specific error.")
    reEvaluatedConfidence: float = Field(..., description="Based on your evaluations score out of 100 on how confident are that this error is present in the sentence and that your thoughts and explanations are valid", ge=0.0, le=100.0)
    errorSpanStart: Optional[int] = Field(..., description="index of first character of the offending span.")
    errorSpanEnd: Optional[int] = Field(..., description="index AFTER the last character  (Python slice convention)")
    errorSpanText: Optional[str] = Field(..., description="exactly mt_string[start:end]  (copy-paste, no changes)")


class AgentOutputStage3(BaseModel):
    consistencyScore: float = Field(..., description="Based on the evaluations of the previous agents, generate a score out of 100 on how consistent the agents are with each other.")
    errorsExists: Literal["NO", "YES"] = Field(..., description="Based on the evaluations of the previous agents i want you to verify whether these errors exists or not. Dont re evaluate. You have to search whether the error flagged by the previous agents exists or not. If it exists then return 'YES' otherwise return 'NO'")
    existanceReasoning: str = Field(..., description="Give brief explanation on your verification of the existance of the errors.")

class CrossReasoningOutput(BaseModel):
    dropped_errors: List[str] = Field(..., description="Subtype errors that should be treated as redundant, unsupported, or dominated by stronger evidence.")
    retained_errors: List[str] = Field(..., description="Subtype errors that remain supported after cross-reasoning.")
    reasoning: str = Field(..., description="Brief explanation of which errors were merged, removed, or kept.")

class AggregationOutput(TypedDict):
    accuracy_error: float
    fluency_error: float
    terminology_error: float
    style_error: float
    overall_error_probability: float
    final_quality_score_100: float

class MTState(TypedDict):
    source: str
    mt: str
    reference: str

    accuracyStage1: Optional[AgentOutputStage1]
    fluencyStage1: Optional[AgentOutputStage1]
    terminologyStage1: Optional[AgentOutputStage1]
    styleStage1: Optional[AgentOutputStage1]
    
    addition: Optional[AgentOutputStage2]
    omission: Optional[AgentOutputStage2]
    mistranslation: Optional[AgentOutputStage2]
    untranslated_text: Optional[AgentOutputStage2]
    transliteration: Optional[AgentOutputStage2]
    non_translation: Optional[AgentOutputStage2]
    punctuation: Optional[AgentOutputStage2]
    spelling: Optional[AgentOutputStage2]
    grammar: Optional[AgentOutputStage2]
    register: Optional[AgentOutputStage2]
    inconsistency: Optional[AgentOutputStage2]
    characterEncoding: Optional[AgentOutputStage2]
    inappropriate_for_context: Optional[AgentOutputStage2]
    inconsistency_use: Optional[AgentOutputStage2]
    awkward: Optional[AgentOutputStage2]

    accuracyStage3: Optional[AgentOutputStage3]
    fluencyStage3: Optional[AgentOutputStage3]
    terminologyStage3: Optional[AgentOutputStage3]
    styleStage3: Optional[AgentOutputStage3]

    accuracyStage1_round: int
    fluencyStage1_round: int
    terminologyStage1_round: int
    styleStage1_round: int

    cross_reasoning: Optional[CrossReasoningOutput]
    aggregation: Optional[AggregationOutput]


llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def make_error_agent_stage1(system_prompt: str, state_key: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
         
        {additional_prompt_for_round}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage1, method="function_calling")

    def agent_fn(state: MTState) -> Dict[str, AgentOutputStage1]:
        if state[state_key + "_round"] == 0:
            state[state_key + "_round"] += 1
            output = chain.invoke({
                "source": state["source"],
                "translated": state["mt"],
                "reference": state["reference"],
                "additional_prompt_for_round": "",
            })
        else:
            state[state_key + "_round"] += 1
            if state_key == "accuracyStage1":
                sub = ["addition", "omission", "mistranslation", "untranslated_text", "transliteration", "non_translation", "accuracyStage3"]
            elif state_key == "fluencyStage1":
                sub = ["punctuation", "spelling", "grammar", "register", "inconsistency", "characterEncoding", "fluencyStage3"]
            elif state_key == "terminologyStage1":
                sub = ["inappropriate_for_context", "inconsistency_use", "terminologyStage3"]
            elif state_key == "styleStage1":
                sub = ["awkward", "styleStage3"]
            round_prompt = f"""
                    You have to re evaluate these sentences again. You are given the outputs of other agents as well.

                    Your Previous Output: {state[state_key]}
                    Other Agents Outputs: {[state[i] for i in sub]}

                    Using these outputs re evaluate these sentence again and re estimate your probabilities and confidence levels.
                    """
            output = chain.invoke({
                "source": state["source"],
                "translated": state["mt"],
                "reference": state["reference"],
                "additional_prompt_for_round": round_prompt,
            })

        return {state_key: output, state_key + "_round": state[state_key + "_round"]}
    return agent_fn

def make_error_agent_stage2(system_prompt: str, state_key: str, SuperCategory: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
        
        PREVIOUS AGENT EVALUATIONS: {previous_agent}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage2, method="function_calling")

    def agent_fn_stage2(state: MTState) -> Dict[str, AgentOutputStage2]:
        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
            "previous_agent": state[SuperCategory]
        })

        return {state_key: output}
    return agent_fn_stage2

def make_error_agent_stage3(system_prompt: str, state_key: str, SuperCategory: str):
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", """
        SOURCE SENTENCE: {source}
         
        MACHINE TRANSLATED SENTENCE: {translated}
         
        REFERENCE SENTENCE: {reference}
        
        SUPER CATEGORY AGENT EVALUATIONS: {previous_agent}
         
        SUB CATEGORY AGENTS EVALUATIONS: {sub_category_agent}
        """)
    ])

    chain = prompt_template | llm.with_structured_output(AgentOutputStage3, method="function_calling")

    def agent_fn_stage3(state: MTState) -> Dict[str, AgentOutputStage3]:
        if SuperCategory == "accuracyStage1":
            sub = ["addition", "omission", "mistranslation", "untranslated_text", "transliteration", "non_translation"]
        elif SuperCategory == "fluencyStage1":
            sub = ["punctuation", "spelling", "grammar", "register", "inconsistency", "characterEncoding"]
        elif SuperCategory == "terminologyStage1":
            sub = ["inappropriate_for_context", "inconsistency_use"]
        elif SuperCategory == "styleStage1":
            sub = ["awkward"]
        
        combined_sub_category = [state[s] for s in sub if state.get(s) is not None]

        output = chain.invoke({
            "source": state["source"],
            "translated": state["mt"],
            "reference": state["reference"],
            "previous_agent": state[SuperCategory],
            "sub_category_agent": combined_sub_category
        })

        return {state_key: output}
    return agent_fn_stage3

def cross_reasoning_node(state: MTState) -> Dict[str, CrossReasoningOutput]:
    ERROR_KEYS = [
    "addition",
    "omission",
    "mistranslation",
    "untranslated_text",
    "transliteration",
    "non_translation",
    "punctuation",
    "spelling",
    "grammar",
    "register",
    "inconsistency",
    "characterEncoding",
    "inappropriate_for_context",
    "inconsistency_use",
    "awkward",
    ]

    cross_reasoning_prompt = ChatPromptTemplate.from_messages([
        ("system", CROSS_RESONING_PROMPT),
        ("human", """
    SOURCE SENTENCE: {source}

    MACHINE TRANSLATED SENTENCE: {translated}

    REFERENCE SENTENCE: {reference}

    SUBTYPE OUTPUTS:
    {subtype_outputs}

    STAGE 3 VERIFICATIONS:
    {stage3_outputs}

    Subtype list:
    {error_keys}
    """
        ),
    ])

    cross_reasoning_chain = cross_reasoning_prompt | llm.with_structured_output(CrossReasoningOutput, method="function_calling")

    subtype_outputs = {
        key: state.get(key).model_dump() if state.get(key) is not None else None
        for key in ERROR_KEYS
    }

    stage3_outputs = {
        "accuracyStage3": state.get("accuracyStage3").model_dump() if state.get("accuracyStage3") is not None else None,
        "fluencyStage3": state.get("fluencyStage3").model_dump() if state.get("fluencyStage3") is not None else None,
        "terminologyStage3": state.get("terminologyStage3").model_dump() if state.get("terminologyStage3") is not None else None,
        "styleStage3": state.get("styleStage3").model_dump() if state.get("styleStage3") is not None else None,
    }

    output = cross_reasoning_chain.invoke({
        "source": state["source"],
        "translated": state["mt"],
        "reference": state["reference"],
        "subtype_outputs": subtype_outputs,
        "stage3_outputs": stage3_outputs,
        "error_keys": ERROR_KEYS,
    })

    return {"cross_reasoning": output}

def should_rerun_accuracy_stage1(state: MTState) -> str:
    if state["accuracyStage1_round"] < 2:
        return "true"
    return "false"

def should_rerun_fluency_stage1(state: MTState) -> str:
    if state["fluencyStage1_round"] < 2:
        return "true"
    return "false"

def should_rerun_terminology_stage1(state: MTState) -> str:
    if state["terminologyStage1_round"] < 2:
        return "true"
    return "false"

def should_rerun_style_stage1(state: MTState) -> str:
    if state["styleStage1_round"] < 2:
        return "true"
    return "false"

def final_sync_node(state: MTState):
    return {}

accuracy_agent = make_error_agent_stage1(ACCURACY_PROMPT, "accuracyStage1")
fluency_agent = make_error_agent_stage1(FLUENCY_PROMPT, "fluencyStage1")
terminology_agent = make_error_agent_stage1(TERMINOLOGY_PROMPT, "terminologyStage1")
style_agent = make_error_agent_stage1(STYLE_PROMPT, "styleStage1")

addition_agent = make_error_agent_stage2(ADDITION_PROMPT, "addition", "accuracyStage1")
omission_agent = make_error_agent_stage2(OMISSION_PROMPT, "omission", "accuracyStage1")
mistranslation_agent = make_error_agent_stage2(MISTRANSLATION_PROMPT, "mistranslation", "accuracyStage1")
untranslated_text_agent = make_error_agent_stage2(UNTRANSLATED_TEXT_PROMPT, "untranslated_text", "accuracyStage1")
transliteration_agent = make_error_agent_stage2(TRANSLITERATION_PROMPT, "transliteration", "accuracyStage1")
non_translation_agent = make_error_agent_stage2(NON_TRANSLATION_PROMPT, "non_translation", "accuracyStage1")
punctuation_agent = make_error_agent_stage2(PUNCTUATION_PROMPT, "punctuation", "fluencyStage1")
spelling_agent = make_error_agent_stage2(SPELLING_PROMPT, "spelling", "fluencyStage1")
grammar_agent = make_error_agent_stage2(GRAMMAR_PROMPT, "grammar", "fluencyStage1")
register_agent = make_error_agent_stage2(REGISTER_PROMPT, "register", "fluencyStage1")
inconsistency_agent = make_error_agent_stage2(INCONSISTENCY_PROMPT, "inconsistency", "fluencyStage1")
characterEncoding_agent = make_error_agent_stage2(CHARACTER_ENCODING_PROMPT, "characterEncoding", "fluencyStage1")
inappropriate_for_context_agent = make_error_agent_stage2(INAPPROPRIATE_FOR_CONTEXT_PROMPT, "inappropriate_for_context", "terminologyStage1")
inconsistency_use_agent = make_error_agent_stage2(INCONSISTENT_USE_PROMPT, "inconsistency_use", "terminologyStage1")
awkward_agent = make_error_agent_stage2(AWKWARD_PROMPT, "awkward", "styleStage1")

accuracy_stage3_agent = make_error_agent_stage3(ACCURACY_STAGE3_PROMPT, "accuracyStage3", "accuracyStage1")
fluency_stage3_agent = make_error_agent_stage3(FLUENCY_STAGE3_PROMPT, "fluencyStage3", "fluencyStage1")
terminology_stage3_agent = make_error_agent_stage3(TERMINOLOGY_STAGE3_PROMPT, "terminologyStage3", "terminologyStage1")
style_stage3_agent = make_error_agent_stage3(STYLE_STAGE3_PROMPT, "styleStage3", "styleStage1")