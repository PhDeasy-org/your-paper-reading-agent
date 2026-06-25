from openai import OpenAI
import instructor
client = instructor.patch(OpenAI(api_key="sk-123"))
print(callable(client.chat.completions))
