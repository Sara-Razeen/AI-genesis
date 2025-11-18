# Old templates - keeping for reference but not using
css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* === GLOBAL STYLES === */
* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: linear-gradient(135deg, #1e3a8a 0%, #1f2937 100%) !important;
}

# .main .block-container {
#     padding-top: 1rem !important;
#     max-width: 900px !important;
#     background: transparent !important;
# }


/* Hide Streamlit branding */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* === HEADER STYLES === */
.header {
    text-align: center;
    padding: 2rem 0;
    margin-bottom: 1rem;
}

.header h1 {
    color: #ffffff;
    font-size: 2.5rem;
    margin-bottom: 0.5rem;
    font-weight: 700;
    text-shadow: 0 2px 4px rgba(0,0,0,0.2);
}

.header p {
    color: rgba(255, 255, 255, 0.8);
    font-size: 1.1rem;
    font-weight: 300;
}

/* Stats cards removed for cleaner design */

/* === CHAT STYLES === */
.chat-message {
    display: flex;
    align-items: flex-start;
    gap: 1rem;
    padding: 1.2rem;
    margin: 1rem 0;
    border-radius: 16px;
}

.chat-message.user {
    background: linear-gradient(135deg, #1e40af 0%, #1f2937 100%);
    color: white;
    margin-left: 2rem;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.3);
}

.chat-message.bot {
    background: rgba(255, 255, 255, 0.95);
    color: #333;
    margin-right: 2rem;
    box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
}              

.avatar {
    width: 40px;
    height: 40px;
    border-radius: 50%;
    overflow: hidden;
    flex-shrink: 0;
}

.avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}

.message {
    flex: 1;
    line-height: 1.6;
}

/* === FOOTER === */
.footer {
    text-align: center;
    color: rgba(255, 255, 255, 0.7);
    font-size: 0.9rem;
    margin-top: 2rem;
    padding: 1.5rem;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 12px;
}

/* === STREAMLIT OVERRIDES === */
.stButton > button {
    background: linear-gradient(135deg, #1e40af 0%, #1f2937 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: 8px !important;
    padding: 0.8rem 1.5rem !important;
    font-weight: 600 !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 15px rgba(30, 64, 175, 0.4) !important;
}

.stTextInput > div > div > input {
    border-radius: 8px !important;
    border: 2px solid rgba(255, 255, 255, 0.3) !important;
    background: rgba(255, 255, 255, 0.9) !important;
    padding: 1rem !important;
}

.stTextInput > div > div > input:focus {
    border-color: #1e40af !important;
    box-shadow: 0 0 0 2px rgba(30, 64, 175, 0.2) !important;
}

/* === RESPONSIVE === */
@media (max-width: 768px) {
    .header h1 {
        font-size: 2rem;
    }
    
    .stats-container {
        flex-direction: column;
        align-items: center;
    }
    
    .chat-message {
        margin-left: 0 !important;
        margin-right: 0 !important;
    }
}
</style>
'''

header_html = '''
<div class="header">
    <h1> AI Resume Screening Assistant</h1>
    <p>Intelligent candidate analysis powered by Gemini AI & Qdrant Vector Search</p>
</div>
'''

stats_html = '''<!-- Stats removed for cleaner design -->'''

bot_template = '''
<div class="chat-message bot">
  <div class="avatar">
    <img src="https://cdn-icons-png.flaticon.com/512/4712/4712101.png" alt="AI Assistant">
  </div>
  <div class="message">
    {{MSG}}
  </div>
</div>
'''

user_template = '''
<div class="chat-message user">
  <div class="avatar">
    <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="You">
  </div>
  <div class="message">
    {{MSG}}
  </div>
</div>
'''

footer_html = '''
<div class="footer">
    <p>© 2025 Resume Screening Assistant | Built with ❤ using Streamlit, Gemini, and Qdrant</p>
</div>
'''