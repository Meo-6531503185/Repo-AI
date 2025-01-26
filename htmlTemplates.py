css = '''
<style>
.chat-container {
    display: flex;
    flex-direction: column;
    margin-bottom: 1.5rem;
}

/* Message container for user */
.chat-container.user {
    align-items: flex-end;
}

/* Message container for bot */
.chat-container.bot {
    align-items: flex-start;
}

/* Shared styles for chat message bubbles */
.chat-message {
    max-width: 70%;
    padding: 1rem;
    border-radius: 1rem;
    margin-bottom: 0.5rem;
    color: #fff;
    font-size: 1rem;
    line-height: 1.5;
}

/* User-specific styles */
.chat-container.user .chat-message {
    background-color: #475063;
    border-bottom-right-radius: 0;
}

/* Bot-specific styles */
.chat-container.bot .chat-message {
    background-color: #2b313e;
    border-bottom-left-radius: 0;
}

/* Avatar container */
.avatar {
    width: 50px;
    height: 50px;
    border-radius: 50%;
    overflow: hidden;
    margin: 0.5rem;
}

/* Avatar image styles */
.avatar img {
    width: 100%;
    height: 100%;
    object-fit: cover;
}'''

bot_template = '''
<div class="chat-container bot">
    <div class="avatar">
        <img src="https://attic.sh/sbzbay0b71yq9x7qjlmxkfex0qmv" alt="Bot Avatar">
    </div>
    <div class="chat-message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-container user">
    <div class="chat-message">{{MSG}}</div>
    <div class="avatar">
        <img src="https://attic.sh/s2vjio0pzc50u1xxtirdfx26nuc5" alt="User Avatar">
    </div>
</div>'''