from openai import OpenAI


class Agent:
    def __init__(self, model, API_KEY):
        self.model = model
        self.llm = OpenAI(api_key = API_KEY)

    def generate_response(self, prompt, context):
        input_text = (
            "Based on the below context, respond with an accurate answer. If you don't find the answer within the context, say I do not know. Don't repeat the question\n\n"
            f"{context}\n\n"
            f"{prompt}"
        )

        response = self.llm.chat.completions.create(
            model = self.model,
            messages =[
                {"role": "user", "content": input_text},
            ],
            max_tokens=150,
            temperature=0
        )
        
        return response.choices[0].message.content.strip()

