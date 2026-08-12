from langgraph.graph import StateGraph, END
from .state import InvoiceState
from .nodes.preprocess import preprocess_node
from .nodes.gemini_extract import gemini_extract_node
from .nodes.extraction import extraction_node
from .nodes.gemini_verification import gemini_verification_node
from .routing import should_verify

def build_graph(enable_gemini: bool = False):
    workflow = StateGraph(InvoiceState)
    
    workflow.add_node("preprocess", preprocess_node)
    workflow.add_node("gemini_extract", gemini_extract_node)
    workflow.add_node("extraction", extraction_node) # this is now the validation node
    
    if enable_gemini:
        workflow.add_node("gemini", gemini_verification_node)
        workflow.add_edge("gemini", "extraction")
    
    workflow.set_entry_point("preprocess")
    workflow.add_edge("preprocess", "gemini_extract")
    workflow.add_edge("gemini_extract", "extraction")
    
    if enable_gemini:
        workflow.add_conditional_edges(
            "extraction",
            should_verify,
            {
                "gemini": "gemini",
                "end": END
            }
        )
    else:
        # Baseline goes straight to END
        workflow.add_edge("extraction", END)
    
    return workflow.compile()
