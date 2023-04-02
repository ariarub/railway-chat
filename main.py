from flask import Flask, jsonify, render_template, request 
import os
import training
import chatbot 

app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    ints = chatbot.predict_class(userText)
    return chatbot.get_response(ints, chatbot.intents)


if __name__ == '__main__':
    app.run(debug=True, port=os.getenv("PORT", default=5000))
