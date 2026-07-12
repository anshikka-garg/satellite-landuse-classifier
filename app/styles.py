import streamlit as st

def inject_premium_styles():
    st.markdown("""
    <style>
        /* Base page styling for dark slate theme */
        html, body, [class*="css"] {
            font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif !important;
        }
        

        

        
        /* Remove default margins and padding for layout alignment */
        .block-container {
            padding-top: 2rem !important;
            padding-bottom: 2rem !important;
        }
    </style>
    """, unsafe_allow_html=True)
