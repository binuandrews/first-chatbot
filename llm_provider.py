import os
from google import genai
from openai import OpenAI

class LLMProvider:

  def __init__(self, config: dict):
    self.provider = config["provider"]
    self.models = config["models"][self.provider]
    if self.provider == "openai":
      key = os.getenv("OPEN_AI_API_KEY")
      if not key:
        raise ValueError("OPEN_AI_API_KEY environment variable is not set")
      self.client = OpenAI(api_key=key)

    elif self.provider == "genai":
      key = os.getenv("GEMINI_API_KEY")
      if not key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
      self.client = genai.Client(api_key=key)
    else:
      raise ValueError(f"Unsupported provider: {self.provider}")
    print(f"{self.provider.title()} client initialized with model: {self.models[0]} 🥃")

  def chat(self, user_message: str) -> str:
    try:
      if self.provider == "openai":
        response = self.client.chat.completions.create(
          model=self.models[0],
          messages=[{"role": "user", "content": user_message}],
          max_tokens=500
        )
        return response.choices[0].message.content.strip()

      elif self.provider == "genai":
        response = self.client.models.generate_content(
          model=self.models[0],
          contents=user_message
        )
        return response.text
    except Exception as e:
      print(f"Error during chat with {self.provider}: {e}")
      return "An error occurred while processing your request."