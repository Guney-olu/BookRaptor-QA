css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

#change the path acc to your ENV (ABS or relative )
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="/Users/aryanrajpurohit/BookRaptor-QA/chat/template/images/robo.jpg" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">
        <p style="font-size: 20px;">{{MSG}}</p>
        <h3 style="color: #3498db; font-size: 14px;">{{TITLE}}</h3>
    </div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="BookRaptor-QA/chat/template/images/robo.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''