from openai import OpenAI

client = OpenAI()


response = client.moderations.create(
    model="omni-moderation-latest",
    input="I want to hurt someone.",
)

print(response)