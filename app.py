from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader
import json
import requests
import gradio as gr
import os

load_dotenv(override=True)


def push(text):
    requests.post(
        "https://api.pushover.net/1/messages.json",
        data={
            "token": os.getenv("PUSHOVER_TOKEN"),
            "user": os.getenv("PUSHOVER_USER"),
            "message": text,
        }
    )


def record_user_details(email, name="Name not provided", notes="not provided"):
    push(f"Recording {name} with email {email} and notes {notes}")
    return {"recorded": "ok"}


def record_unknown_question(question):
    push(f"Recording {question}")
    return {"recorded": "ok"}


record_user_details_json = {
    "name": "record_user_details",
    "description": "Use this tool to record that a user is interested in being in touch and provided an email address",
    "parameters": {
        "type": "object",
        "properties": {
            "email": {
                "type": "string",
                "description": "The email address of this user"
            },
            "name": {
                "type": "string",
                "description": "The user's name, if they provided it"
            },
            "notes": {
                "type": "string",
                "description": "Any additional information about the conversation that's worth recording to give context"
            }
        },
        "required": ["email"],
        "additionalProperties": False
    }
}

record_unknown_question_json = {
    "name": "record_unknown_question",
    "description": "Always use this tool to record any question that couldn't be answered as you didn't know the answer",
    "parameters": {
        "type": "object",
        "properties": {
            "question": {
                "type": "string",
                "description": "The question that couldn't be answered"
            },
        },
        "required": ["question"],
        "additionalProperties": False
    }
}

tools = [{"type": "function", "function": record_user_details_json},
         {"type": "function", "function": record_unknown_question_json}]


class Chatbot:
    def __init__(self):
        self.openai = OpenAI(api_key=os.getenv("GOOGLE_API_KEY"),
                             base_url="https://generativelanguage.googleapis.com/v1beta/openai/")
        self.name = "Alan Hurtarte"
        self.linkedin = ""
        self.summary = ""

        self.init_summary()
        self.init_linkedin()

    def get_summary(self):
        return self.summary

    def get_linkedin(self):
        return self.linkedin

    def get_name(self):
        return self.name

    def init_summary(self):
        with open("me/summary.txt", "r", encoding="utf-8") as f:
            self.summary = f.read()

    def init_linkedin(self):
        self.linkedin = PdfReader("me/linkedin.pdf")

    def handle_tool_call(self, tool_calls):
        results = []
        for tool_call in tool_calls:
            arguments = json.loads(tool_call.function.arguments)
            if tool_call.function.name == "record_user_details":
                result = record_user_details(**arguments)
            elif tool_call.function.name == "record_unknown_question":
                result = record_unknown_question(**arguments)

            results.append({"role": "tool", "content": json.dumps(
                result), "tool_call_id": tool_call.id})
        return results

    def system_prompt(self):
        system_prompt = f"You are acting as {self.name}. You are answering questions on {self.name}'s website, \
particularly questions related to {self.name}'s career, background, skills and experience. \
Your responsibility is to represent {self.name} for interactions on the website as faithfully as possible. \
You are given a summary of {self.name}'s background and LinkedIn profile which you can use to answer questions. \
Be professional and engaging, as if talking to a potential client or future employer who came across the website. \
If you don't know the answer to any question, use your record_unknown_question tool to record the question that you couldn't answer, even if it's about something trivial or unrelated to career. \
If the user is engaging in discussion, try to steer them towards getting in touch via email; ask for their email and record it using your record_user_details tool. "

        system_prompt += f"\n\n## Summary:\n{self.summary}\n\n## LinkedIn Profile:\n{self.linkedin}\n\n"
        system_prompt += f"With this context, please chat with the user, always staying in character as {self.name}."
        return system_prompt

    def chat(self, message, history):
        messages = [{"role": "system", "content": self.system_prompt(
        )}] + history + [{"role": "user", "content": message}]
        done = False
        while not done:
            response = self.openai.chat.completions.create(
                model="gemini-2.0-flash", messages=messages, tools=tools)
            if response.choices[0].finish_reason == "tool_calls":
                message = response.choices[0].message
                tool_calls = message.tool_calls
                results = self.handle_tool_call(tool_calls)
                messages.append(message)
                messages.extend(results)
            else:
                done = True
        return response.choices[0].message.content


if __name__ == "__main__":
    me = Chatbot()

    # Custom CSS for styling
    custom_css = """
    .gradio-container {
        max-width: 800px !important;
        margin: auto !important;
    }
    .chat-message {
        padding: 1.5rem !important;
        border-radius: 0.5rem !important;
        margin-bottom: 1rem !important;
        display: flex !important;
        flex-direction: column !important;
    }
    .chat-message.user {
        background-color: #f0f7ff !important;
        border: 1px solid #e0e7ff !important;
    }
    .chat-message.bot {
        background-color: #f8fafc !important;
        border: 1px solid #e2e8f0 !important;
    }
    .chat-message .message {
        font-size: 1rem !important;
        line-height: 1.5 !important;
    }
    .chat-message .avatar {
        width: 40px !important;
        height: 40px !important;
        border-radius: 50% !important;
        margin-right: 1rem !important;
    }
    .chat-input {
        border-radius: 0.5rem !important;
        padding: 1rem !important;
        border: 1px solid #e2e8f0 !important;
    }
    .chat-input:focus {
        border-color: #3b82f6 !important;
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.1) !important;
    }
    .submit-button {
        background-color: #3b82f6 !important;
        color: white !important;
        border-radius: 0.5rem !important;
        padding: 0.75rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.2s !important;
    }
    .submit-button:hover {
        background-color: #2563eb !important;
        transform: translateY(-1px) !important;
    }
    """

    # Create the interface with custom styling
    interface = gr.ChatInterface(
        me.chat,
        type="messages",
        title="Alan Hurtarte's Chatbot",
        description="Ask Alan Hurtarte anything about his career, background, skills and experience.",
        theme=gr.themes.Soft(
            primary_hue="blue",
            secondary_hue="blue",
            neutral_hue="slate",
            radius_size="md",
            font=["Inter", "ui-sans-serif", "system-ui", "sans-serif"],
        ),
        css=custom_css,
    )

    interface.launch()
