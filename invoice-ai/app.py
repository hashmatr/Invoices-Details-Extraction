import os
import streamlit as st
from config.settings import settings
import time
import tempfile
from graph.graph import build_graph
from models.metrics import ProcessingMetrics

st.set_page_config(
    page_title="Intelligent Invoice Extraction", 
    layout="wide"
)

# Inject custom CSS for premium blue theme and SVG alignment
st.markdown("""
<style>
    /* Primary Theme Colors */
    :root {
        --primary-blue: #1E3A8A;
        --secondary-blue: #3B82F6;
        --accent-blue: #DBEAFE;
    }
    
    .stApp {
        background-color: #F8FAFC;
    }
    
    h1, h2, h3 {
        color: #1E3A8A !important;
        font-family: 'Inter', sans-serif;
        display: flex;
        align-items: center;
    }
    
    .stButton>button {
        background-color: #3B82F6 !important;
        color: white !important;
        border-radius: 8px !important;
        border: none !important;
        padding: 0.5rem 1rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
    }
    
    .stButton>button:hover {
        background-color: #2563EB !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06) !important;
    }
    
    /* Metric Cards */
    div[data-testid="stMetricValue"] {
        color: #1E3A8A !important;
        font-weight: 700 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #64748B !important;
        font-weight: 500 !important;
        font-size: 1rem !important;
    }
    
    /* Upload Widget Styling */
    .stFileUploader {
        border-radius: 12px;
        padding: 1rem;
        background-color: white;
        border: 2px dashed #93C5FD;
    }
    
    .svg-icon {
        vertical-align: middle;
        margin-right: 12px;
    }
    
    .info-container {
        display: flex;
        align-items: center;
        background-color: #DBEAFE;
        padding: 1rem;
        border-radius: 8px;
        color: #1E3A8A;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Main Title
st.markdown("""
    <h1>
        <svg class="svg-icon" xmlns="http://www.w3.org/2000/svg" width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"></path><polyline points="14 2 14 8 20 8"></polyline><line x1="16" y1="13" x2="8" y2="13"></line><line x1="16" y1="17" x2="8" y2="17"></line><polyline points="10 9 9 9 8 9"></polyline></svg>
        Intelligent Invoice Extraction
    </h1>
""", unsafe_allow_html=True)
st.markdown("Automatically extract highly accurate, structured data from Chinese invoices.")

# Move Uploader to Sidebar
with st.sidebar:
    st.markdown("""
        <h3>
            <svg class="svg-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 19a2 2 0 0 1-2 2H4a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h5l2 3h9a2 2 0 0 1 2 2z"></path></svg>
            Upload Document
        </h3>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload Chinese Invoice", type=["pdf", "png", "jpg", "jpeg"], help="Supported formats: PDF, PNG, JPG")
    
    process_button = False
    if uploaded_file is not None:
        process_button = st.button("Process Invoice", use_container_width=True)

if uploaded_file is not None and process_button:
    # Save uploaded file to temporary location
    with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name
        
    file_type = "pdf" if uploaded_file.name.lower().endswith(".pdf") else "image"
    
    # Initialize Graph (Gemini intentionally disabled behind the scenes)
    app = build_graph(enable_gemini=False)
    initial_state = {
        "file_path": tmp_file_path,
        "file_type": file_type,
        "images": [],
        "ocr_data": [],
        "layout_data": {},
        "qr_code_present": False,
        "qr_data": [],
        "invoice_data": None,
        "validation_errors": [],
        "suspicious_fields": [],
        "gemini_corrections": {},
        "metrics": ProcessingMetrics()
    }
    
    with st.spinner("Extracting data from invoice..."):
        # Execute workflow silently
        result_state = None
        for event in app.stream(initial_state):
            for node_name, node_state in event.items():
                result_state = node_state
        
    st.success("Extraction Complete!")
    
    invoice = result_state.get("invoice_data")
    
    # Display Results
    if invoice:
        st.markdown("""
            <h3>
                <svg class="svg-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="18" y1="20" x2="18" y2="10"></line><line x1="12" y1="20" x2="12" y2="4"></line><line x1="6" y1="20" x2="6" y2="14"></line></svg>
                Extracted Details
            </h3>
        """, unsafe_allow_html=True)
        st.markdown("---")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Invoice Number", invoice.invoice_number.value if invoice.invoice_number else "N/A")
            st.metric("Invoice Date", str(invoice.invoice_date.value) if invoice.invoice_date else "N/A")
        with col2:
            st.metric("Supplier", invoice.supplier.name.value if invoice.supplier and invoice.supplier.name else "N/A")
            st.metric("Customer", invoice.customer.name.value if invoice.customer and invoice.customer.name else "N/A")
        with col3:
            st.metric("Amount Before VAT", f"¥ {invoice.amount_before_vat.value}" if invoice.amount_before_vat else "N/A")
            st.metric("Total Amount", f"¥ {invoice.total_amount_after_vat.value}" if invoice.total_amount_after_vat else "N/A")
            
    errors = result_state.get("validation_errors", [])
    if errors:
        st.markdown("""
            <div style="background-color: #FEF3C7; color: #92400E; padding: 1rem; border-radius: 8px; margin-bottom: 1rem; display: flex; align-items: center;">
                <svg class="svg-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"></path><line x1="12" y1="9" x2="12" y2="13"></line><line x1="12" y1="17" x2="12.01" y2="17"></line></svg>
                Some fields require manual review.
            </div>
        """, unsafe_allow_html=True)
        for err in errors:
            st.markdown(f"- **{err['field']}**: {err['message']}")
            
    # Cleanup temp file
    os.unlink(tmp_file_path)
    
elif uploaded_file is None:
    st.markdown("""
        <div class="info-container">
            <svg class="svg-icon" xmlns="http://www.w3.org/2000/svg" width="24" height="24" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="19" y1="12" x2="5" y2="12"></line><polyline points="12 19 5 12 12 5"></polyline></svg>
            Please upload an invoice document from the sidebar to begin.
        </div>
    """, unsafe_allow_html=True)
