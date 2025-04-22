from flask import Flask, request, jsonify
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

app = Flask(__name__)

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=GOOGLE_API_KEY)

# System instructions in both English and Arabic
system_instructions = """
You are an Islamic scholar assistant specialized exclusively in matters related to:
- Hajj (الحج)
- Umrah (العمرة)
- Islamic rituals (العبادات الإسلامية)
- Related fiqh (الفقه المتعلق بالحج والعمرة)
- Islamic ethics (الأخلاق الإسلامية)

Your responsibilities:
1. Only answer questions about Hajj, Umrah, and directly related Islamic matters
2. Provide authentic information based on Quran and Sunnah
3. Clearly state if something is an opinion from a specific madhab
4. Refuse to answer any questions outside your scope politely
5. Respond in the same language the question was asked in (Arabic or English)
6. For Arabic responses, use proper Islamic terminology (المصطلحات الشرعية)
7. Keep answers concise but comprehensive

If asked about other topics, respond with:
English: "I specialize only in Hajj, Umrah, and related Islamic matters. Please ask about those topics."
Arabic: "أنا متخصص فقط في أمور الحج والعمرة وما يتعلق بهما من أمور إسلامية. الرجاء السؤال في هذه المواضيع."
"""

# Create the model
model = genai.GenerativeModel(
    'gemini-2.0-flash',
    system_instruction=system_instructions,
    generation_config={
        "temperature": 0.5,
        "top_p": 1,
        "top_k": 32,
        "max_output_tokens": 1024,
    },
    safety_settings=[
        {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
        {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
    ]
)

@app.route('/chat', methods=['POST'])
def chat():
    data = request.get_json()
    user_input = data.get('message', '')
    chat_session = model.start_chat(history=[])
    
    try:
        response = chat_session.send_message(user_input)
        return jsonify({
            'response': response.text,
            'status': 'success'
        })
    except Exception as e:
        return jsonify({
            'response': str(e),
            'status': 'error'
        }), 500

@app.route('/')
def home():
    return "Hajj & Umrah Chatbot API is running. Use POST /chat endpoint to interact."

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
