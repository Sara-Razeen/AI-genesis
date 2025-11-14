# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#   width: 20%;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# css = '''
# <style>
# .chat-message {
#     padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
# }
# .chat-message.user {
#     background-color: #2b313e
# }
# .chat-message.bot {
#     background-color: #475063
# }
# .chat-message .avatar {
#   width: 20%;
# }
# .chat-message .avatar img {
#   max-width: 78px;
#   max-height: 78px;
#   border-radius: 50%;
#   object-fit: cover;
# }
# .chat-message .message {
#   width: 80%;
#   padding: 0 1.5rem;
#   color: #fff;
# }
# '''

# bot_template = '''
# <div class="chat-message bot">
#     <div class="avatar">
#         <img src="https://i.ibb.co/cN0nmSj/Screenshot-2023-05-28-at-02-37-21.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
#     </div>
#     <div class="message">{{MSG}}</div>
# </div>
# '''

# user_template = '''
# <div class="chat-message user">
#     <div class="avatar">
#         <img src="https://i.ibb.co/rdZC7LZ/Photo-logo-1.png">
#     </div>    
#     <div class="message">{{MSG}}</div>
# </div>
# '''
css = '''
<style>
body {
    background: linear-gradient(145deg, #f8faff, #eef2ff);
    font-family: 'Inter', sans-serif;
    color: #1e293b;
}

/* === HEADER === */
.header {
    text-align: center;
    padding: 2rem 0 1rem 0;
}
.header h1 {
    font-size: 2rem;
    color: #2563eb;
    margin-bottom: 0.4rem;
    font-weight: 700;
}
.header p {
    color: #64748b;
    font-size: 0.95rem;
}

/* === DASHBOARD STATS === */
.stats-container {
    display: flex;
    justify-content: center;
    gap: 1rem;
    margin-bottom: 2rem;
}
.stat-card {
    background: white;
    border-radius: 12px;
    padding: 1rem 1.5rem;
    box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    width: 270px;
    transition: 0.25s ease-in-out;
    border: 1px solid #e2e8f0;
}
.stat-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 6px 16px rgba(59,130,246,0.2);
}
.stat-card h3 {
    color: #334155;
    font-size: 1rem;
    margin-bottom: 0.3rem;
}
.stat-card .value {
    font-size: 1.8rem;
    font-weight: 700;
    color: #2563eb;
}
.stat-card .desc {
    color: #10b981;
    font-size: 0.85rem;
    margin-top: 0.2rem;
}

/* === CHAT AREA === */
.chat-container {
    width: 90%;
    max-width: 850px;
    margin: 0 auto 3rem auto;
}
.chat-message {
    display: flex;
    align-items: flex-start;
    border-radius: 14px;
    padding: 1rem 1.3rem;
    margin-bottom: 1rem;
    transition: 0.3s ease-in-out;
    box-shadow: 0 2px 10px rgba(0,0,0,0.1);
}
.chat-message.user {
    background: #eff6ff;
    border: 1px solid #93c5fd;
}
.chat-message.bot {
    background: #ffffff;
    border: 1px solid #cbd5e1;
}
.chat-message .avatar {
    width: 55px;
    height: 55px;
    border-radius: 50%;
    overflow: hidden;
    margin-right: 1.2rem;
}
.chat-message .avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}
.chat-message .message {
    flex-grow: 1;
    color: #1e293b;
    font-size: 0.95rem;
    line-height: 1.6;
}

/* === FOOTER === */
.footer {
    text-align: center;
    font-size: 0.85rem;
    color: #94a3b8;
    margin-top: 2rem;
    padding-bottom: 1rem;
}
</style>
'''

header_html = '''
<div class="header">
    <h1>💼 Resume Screening Assistant</h1>
    <p>Chat with your uploaded resumes or PDFs — powered by Gemini + Qdrant</p>
</div>
'''

stats_html = '''
<div class="stats-container">
    <div class="stat-card">
        <h3>Total Resumes</h3>
        <div class="value">0</div>
        <div class="desc">+0% from last week</div>
    </div>
    <div class="stat-card">
        <h3>Candidates Screened</h3>
        <div class="value">0</div>
        <div class="desc">Ready to review</div>
    </div>
    <div class="stat-card">
        <h3>Match Rate</h3>
        <div class="value">0%</div>
        <div class="desc">Avg. qualification score</div>
    </div>
</div>
'''

bot_template = '''
<div class="chat-container">
  <div class="chat-message bot">
    <div class="avatar">
      <img src="https://cdn-icons-png.flaticon.com/512/4712/4712101.png" alt="Bot Avatar">
    </div>
    <div class="message">{{MSG}}</div>
  </div>
</div>
'''

user_template = '''
<div class="chat-container">
  <div class="chat-message user">
    <div class="avatar">
      <img src="https://cdn-icons-png.flaticon.com/512/3135/3135715.png" alt="User Avatar">
    </div>
    <div class="message">{{MSG}}</div>
  </div>
</div>
'''

footer_html = '''
<div class="footer">
    <p>© 2025 Resume Screening Assistant | Built with ❤ using Streamlit, Gemini, and Qdrant</p>
</div>
'''