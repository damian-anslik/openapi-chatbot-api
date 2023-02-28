import openai
import numpy as np
from os import environ
from time import time
from tinydb import TinyDB, where
from fastapi import FastAPI, status
from fastapi.exceptions import RequestValidationError

openai.api_key = environ.get("API_KEY")
openai.organization = environ.get("ORGANIZATION_ID")
app = FastAPI()
db = TinyDB("db.json")
training_db = TinyDB("training.json")
contributors_db = TinyDB("contributors.json")

def calculate_prompt_embedding(prompt: str) -> np.ndarray:
    """ 
    Calculates the embedding for a given prompt.  
    """
    embedding = openai.Embedding.create(
        model="text-embedding-ada-002",
        input=prompt
    )
    return np.array(embedding["data"][0]["embedding"])

def calculate_similarity(embedding1: np.ndarray, embedding2: np.ndarray) -> float:
    """  
    Calculates the cosine similarity between two embeddings.
    """
    return np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))

def is_valid_prompt(prompt: str) -> bool:
    """  
    Checks if the prompt is valid. A prompt is valid if it is similar to any of the prompts in the training database.
    """
    prompt_embedding = calculate_prompt_embedding(prompt)
    for response in training_db.all():
        response_embedding = calculate_prompt_embedding(response["prompt"])
        similarity = calculate_similarity(prompt_embedding, response_embedding)
        print(similarity)
        if similarity > 0.80:
            return True
    return False

@app.get("/completion", status_code=status.HTTP_200_OK)
async def request_completion(prompt: str):
    """
    Returns a completion for the given prompt. The completion is stored in the database for future reference.
    """
    if not is_valid_prompt(prompt):
        return {"text": "Invalid prompt", "id": None}
    response = openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        temperature=0,
        max_tokens=500,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0
    )
    response_data = {"text": response["choices"][0]["text"], "id": response["id"]}
    db.insert({"prompt": prompt, "score": 0, **response})
    return response_data

@app.post("/completion/feedback", status_code=status.HTTP_200_OK)
async def provide_completion_feedback(id: str, score: int, feedback: str = None):
    """  
    User provides feedback on the completion. The feedback is stored for future reference.
    """
    response = db.get(where("id") == id)
    if not response:
        raise RequestValidationError
    response["score"] = score
    response["feedback"] = feedback
    db.update(response, where("id") == id)

@app.post("/completion/contribute", status_code=status.HTTP_200_OK)
async def add_new_training_sample(prompt: str, completion: str, contributor_token: str):
    """  
    Inserts a training example into the training database. A job will be scheduled to train the model with the new data.
    """
    contributor = contributors_db.get(where("token") == contributor_token)
    if not contributor:
        raise RequestValidationError
    # We can perform some validation here
    training_db.insert({"prompt": prompt, "completion": completion, "timestamp": time(), "contributor": contributor_token})