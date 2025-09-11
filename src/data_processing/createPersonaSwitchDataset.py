from google import genai
from itertools import permutations
import os


client = genai.Client(api_key="AIzaSyDZZq12UYpYjiORY8tmAgDhtrrz4WO5vRY")

ageCategories = ["child", "adolescent", "adult", "older adult"]

for first, second in permutations(ageCategories, 2):
    for i in range(4):
        # The user starts as 'first', then switches to 'second'
        prompt = (
            f"Generate a 5 turn conversation between a human user and an ai assistant, following the ### Human: and ### Assistant: format. For the first two turns, the user is a {first}. At the beginning of the third turn, switch the user person to someone aged {second} but do not say anything about it. For the next three turns, the user is a {second}. Make sure to follow the persona of the user and the assistant. So, in total, there should be 5 turns, each with one human message and one assistant message, with the user being {first} for the first two turns and {second} for the next three turns."
        )
        response = client.models.generate_content(
            model="gemini-2.5-pro",
            contents=prompt,
        )
        # Extract the actual content from the response
        conversation = response.text
        #put the conversation into a file
        with open(f"personaSwitchDSSecret/conversation_{first}_{second}_{i}.txt", "w") as f:
            f.write(conversation)
        print(f"Conversation {first}_{second}_{i} created")

'''
for i in range(1, 4):
    first = "older adult"
    second = "adult"
    # The user starts as 'first', then switches to 'second'
    prompt = (
        f"Generate a 5 turn conversation between a human user and an ai assistant, following the ### Human: and ### Assistant: format. For the first two turns, the user is a {first}. At the beginning of the third turn, the user should say they are leaving and now giving control to a {second}. For the next three turns, the user is a {second}. Make sure to follow the persona of the user and the assistant."
    )
    response = client.models.generate_content(
        model="gemini-2.5-pro",
        contents=prompt,
    )
    # Extract the actual content from the response
    conversation = response.text
    #put the conversation into a file
    with open(f"personaSwitchDS/conversation_{first}_{second}_{i}.txt", "w") as f:
        f.write(conversation)
'''
#Gemini API: AIzaSyDZZq12UYpYjiORY8tmAgDhtrrz4WO5vRY
#OpenAI API: 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
