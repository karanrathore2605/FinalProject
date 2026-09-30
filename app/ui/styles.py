"""
UI Styling module for Streamlit interface.
Provides custom CSS for high-visibility typography, modern chat bubbles,
clean document status indicators, and responsive controls.
"""
import streamlit as st

CUSTOM_CSS = """
<style>
/* Global typography enhancement */
html, body, [class*="css"], .stMarkdown {
    font-size: 18px;
}

/* Chat message bubble styling & spacing */
[data-testid="stChatMessage"] {
    background-color: #f8f9fa;
    border-radius: 12px;
    padding: 16px 20px;
    margin-bottom: 16px;
    border: 1px solid #edf0f2;
}

/* Assistant and user response text */
[data-testid="stChatMessage"] p,
[data-testid="stChatMessage"] li,
[data-testid="stChatMessage"] span,
[data-testid="stChatMessageContent"] p,
[data-testid="stChatMessageContent"] li,
[data-testid="stChatMessageContent"] span,
[data-testid="stChatMessageContent"] div {
    font-size: 20px !important;
    line-height: 1.65 !important;
    color: #1e293b !important;
}

/* Tables inside assistant answers */
[data-testid="stChatMessage"] table,
[data-testid="stChatMessageContent"] table {
    font-size: 19px !important;
    width: 100% !important;
    margin: 14px 0 !important;
}
[data-testid="stChatMessage"] th,
[data-testid="stChatMessageContent"] th {
    font-size: 20px !important;
    font-weight: 600 !important;
    padding: 12px 16px !important;
    background-color: #f1f5f9 !important;
}
[data-testid="stChatMessage"] td,
[data-testid="stChatMessageContent"] td {
    font-size: 19px !important;
    padding: 10px 16px !important;
}

/* Tool used caption */
.tool-caption {
    color: #475569;
    font-size: 16px !important;
    font-weight: 600 !important;
    margin-bottom: 6px;
    margin-top: 18px;
    font-family: inherit;
    letter-spacing: 0.2px;
}

/* Document ready card in sidebar */
.doc-ready-card {
    background-color: #e8f5e9;
    border: 1px solid #c8e6c9;
    border-radius: 8px;
    padding: 14px 16px;
    color: #1b5e20;
    font-weight: 600;
    font-size: 18px !important;
    margin: 14px 0 10px 0;
}

/* Chat input box (st.chat_input) */
[data-testid="stChatInput"] textarea {
    font-size: 20px !important;
    line-height: 1.5 !important;
    font-family: inherit !important;
    min-height: 58px !important;
    padding: 12px 16px !important;
}
[data-testid="stChatInput"] textarea::placeholder {
    font-size: 20px !important;
    color: #94a3b8 !important;
}

.stTextInput input, div[data-testid="stTextInput"] input {
    font-size: 20px !important;
}

/* User question text inside chat feed */
.user-question-text {
    font-size: 21px !important;
    font-weight: 600 !important;
    line-height: 1.5 !important;
    color: #0f172a !important;
}

/* Sidebar controls & labels */
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] .stHeading {
    font-size: 26px !important;
}
[data-testid="stSidebar"] p, 
[data-testid="stSidebar"] span, 
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] div {
    font-size: 17px !important;
}
[data-testid="stSidebar"] .stCaption {
    font-size: 15px !important;
    color: #64748b !important;
}

/* Buttons */
.stButton button,
[data-testid="stBaseButton-secondary"],
[data-testid="stBaseButton-primary"] {
    font-size: 17px !important;
    font-weight: 500 !important;
    padding: 8px 16px !important;
}

/* Expanders (Sources) */
[data-testid="stExpander"] details summary p {
    font-size: 18px !important;
    font-weight: 600 !important;
}
[data-testid="stExpander"] details div p, 
[data-testid="stExpander"] details div span,
[data-testid="stExpander"] details div .stCaption {
    font-size: 16px !important;
}
</style>
"""


def inject_custom_css() -> None:
    """Inject the custom CSS into the Streamlit app head."""
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
