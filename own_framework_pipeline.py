from own_framework import *
from aggregation import aggregate_mt_quality
from langgraph.graph import START, END
import json

graph = StateGraph(MTState)

graph.add_node("accuracyStage1_node", accuracy_agent)
graph.add_node("fluencyStage1_node", fluency_agent)
graph.add_node("terminologyStage1_node", terminology_agent)
graph.add_node("styleStage1_node", style_agent)
graph.add_node("addition_node", addition_agent)
graph.add_node("omission_node", omission_agent)
graph.add_node("mistranslation_node", mistranslation_agent)
graph.add_node("untranslated_text_node", untranslated_text_agent)
graph.add_node("transliteration_node", transliteration_agent)
graph.add_node("non_translation_node", non_translation_agent)
graph.add_node("punctuation_node", punctuation_agent)
graph.add_node("spelling_node", spelling_agent)
graph.add_node("grammar_node", grammar_agent)
graph.add_node("register_node", register_agent)
graph.add_node("inconsistency_node", inconsistency_agent)
graph.add_node("characterEncoding_node", characterEncoding_agent)
graph.add_node("inappropriate_for_context_node", inappropriate_for_context_agent)
graph.add_node("inconsistency_use_node", inconsistency_use_agent)
graph.add_node("awkward_node", awkward_agent)
graph.add_node("accuracyStage3_node", accuracy_stage3_agent)
graph.add_node("fluencyStage3_node", fluency_stage3_agent)
graph.add_node("terminologyStage3_node", terminology_stage3_agent)
graph.add_node("styleStage3_node", style_stage3_agent)
graph.add_node("final_sync_node", final_sync_node)
graph.add_node("cross_reasoning_node", cross_reasoning_node)
graph.add_node("aggregation_node", aggregate_mt_quality)

# graph.set_entry_point("START")

graph.add_edge(START, "accuracyStage1_node")
graph.add_edge(START, "fluencyStage1_node")
graph.add_edge(START, "terminologyStage1_node")
graph.add_edge(START, "styleStage1_node")

graph.add_edge("accuracyStage1_node", "addition_node")
graph.add_edge("accuracyStage1_node", "omission_node")
graph.add_edge("accuracyStage1_node", "mistranslation_node")
graph.add_edge("accuracyStage1_node", "untranslated_text_node")
graph.add_edge("accuracyStage1_node", "transliteration_node")
graph.add_edge("accuracyStage1_node", "non_translation_node")

graph.add_edge("fluencyStage1_node", "punctuation_node")
graph.add_edge("fluencyStage1_node", "spelling_node")
graph.add_edge("fluencyStage1_node", "grammar_node")
graph.add_edge("fluencyStage1_node", "register_node")
graph.add_edge("fluencyStage1_node", "inconsistency_node")
graph.add_edge("fluencyStage1_node", "characterEncoding_node")

graph.add_edge("terminologyStage1_node", "inappropriate_for_context_node")
graph.add_edge("terminologyStage1_node", "inconsistency_use_node")

graph.add_edge("styleStage1_node", "awkward_node")

graph.add_edge("addition_node", "accuracyStage3_node")
graph.add_edge("omission_node", "accuracyStage3_node")
graph.add_edge("mistranslation_node", "accuracyStage3_node")
graph.add_edge("untranslated_text_node", "accuracyStage3_node")
graph.add_edge("transliteration_node", "accuracyStage3_node")
graph.add_edge("non_translation_node", "accuracyStage3_node")

graph.add_edge("punctuation_node", "fluencyStage3_node")
graph.add_edge("spelling_node", "fluencyStage3_node")
graph.add_edge("grammar_node", "fluencyStage3_node")
graph.add_edge("register_node", "fluencyStage3_node")
graph.add_edge("inconsistency_node", "fluencyStage3_node")
graph.add_edge("characterEncoding_node", "fluencyStage3_node")

graph.add_edge("inappropriate_for_context_node", "terminologyStage3_node")
graph.add_edge("inconsistency_use_node", "terminologyStage3_node")

graph.add_edge("awkward_node", "styleStage3_node")

graph.add_conditional_edges("accuracyStage3_node", should_rerun_accuracy_stage1, {"true": "accuracyStage1_node", "false": "final_sync_node"})
graph.add_conditional_edges("fluencyStage3_node", should_rerun_fluency_stage1, {"true": "fluencyStage1_node", "false": "final_sync_node"})
graph.add_conditional_edges("terminologyStage3_node", should_rerun_terminology_stage1, {"true": "terminologyStage1_node", "false": "final_sync_node"})
graph.add_conditional_edges("styleStage3_node", should_rerun_style_stage1, {"true": "styleStage1_node", "false": "final_sync_node"})

# graph.add_edge("accuracyStage3_node", "aggregation_node")
# graph.add_edge("fluencyStage3_node", "aggregation_node")
# graph.add_edge("terminologyStage3_node", "aggregation_node")
# graph.add_edge("styleStage3_node", "aggregation_node")

graph.add_edge("final_sync_node", "cross_reasoning_node")
graph.add_edge("cross_reasoning_node", "aggregation_node")
graph.add_edge("aggregation_node", END)

app = graph.compile()

def serialize_state(obj):
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    elif isinstance(obj, dict):
        return {k: serialize_state(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_state(v) for v in obj]
    else:
        return obj

if __name__ == "__main__":
    intput_state = {
        "source": "The qualities that determine a subculture as distinct may be linguistic, aesthetic, religious, political, sexual, geographical, or a combination of factors.",
        "mt": "वे गुण जो किसी उप-संस्कृति को अलग बनाते हैं, जैसे कि भाषा, सौंदर्य, धर्म, राजनीति, यौन, भूगोल या कई सारे कारकों का मिश्रण हो सकते हैं.",
        "reference": "उपसंस्कृति को विशिष्ट रूप से निर्धारित करने वाले गुण भाषाई, सौंदर्य, धार्मिक, राजनीतिक, यौन, भौगोलिक या कारकों का संयोजन हो सकते हैं।",
        "accuracyStage1_round": 0,
        "fluencyStage1_round": 0,
        "terminologyStage1_round": 0,
        "styleStage1_round": 0,
    }

    result = app.invoke(intput_state)

    serialized_result = serialize_state(result)

    with open("result.json", "w", encoding="utf-8") as f:
        json.dump(serialized_result, f, indent=4, ensure_ascii=False)

    print("Saved to result.json")