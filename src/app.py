from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import jsonlines
import json
import os

app = FastAPI()

class RequestJsonInsertion(BaseModel):
    context: str
    qas: list


@app.post("/insertdata")
def insert_json_line(request: RequestJsonInsertion):
    """
    Creates a Question-Answering model based on the model parameters and
    takes a context to generate answers with their confidence.
    Parameters
    ----------
    ``context`` : str <br>
        - Context of the text as a string. <br>
    ``qas`` : str <br>
        - Json list of Question and answering. See further documentation below. <br>

    Returns
    -------
    int <br>
        - Inserts into trainsamples and returns success as message.

    Examples request
    --------
    <pre>
    {
    "context": "Hey, I am Oweys. I am a Consultant at IBM.", 
    "qas": [
        {"question": "Who is Oweys?", 
        "answers": ["a consultant at IBM."]}
        ]
    }
    </pre>
    """
        
    error_handling_jsonl(request)
    # get absolute path
    trainset_name = 'trainsamples.jsonl'
    data_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data', trainset_name))
    request_dict = request.dict()

    with jsonlines.open(data_path, mode='r') as reader:
        lines = list(reader)

    # insert the new line at the end of the file
    lines.insert(len(lines), request_dict)

    with jsonlines.open(data_path, mode='w') as writer:
        writer.write_all(lines)

    return {"message": "JSON line inserted successfully."}


@app.get("/runtraining")
def run_training():
    """
    Runs the training on the current available trainsamples and overrides the old model if the new model outperformed.

    <span style="color:red; font-size: 2em;">THIS IS STILL IN ACTIVE DEVELOPMENT</span>
    """
    new_f1_score = 80
    old_f1_score = 79

    return {"message": f"Training successful. Outperforming results: New F1 score-{new_f1_score} & Old F1 score-{old_f1_score}"}


@app.post("/predict")
def predict():
    """
    Predicts based on a context and a question.

    <span style="color:red; font-size: 2em;">THIS IS STILL IN ACTIVE DEVELOPMENT</span>
    """

    return {"Answers": False}


def error_handling_jsonl(request : RequestJsonInsertion):

    try:
        data = json.loads(request.json())
        if not isinstance(data, dict):
            raise HTTPException(status_code=400, detail="JSON object expected")
        if "context" not in data:
            raise HTTPException(status_code=400, detail="Missing 'context' field")
        if not isinstance(data["context"], str):
            raise HTTPException(status_code=400, detail="'context' field should be a string")
        if "qas" not in data:
            raise HTTPException(status_code=400, detail="Missing 'qas' field")
        qas = data["qas"]

        if not isinstance(qas, list):
            raise HTTPException(status_code=400, detail="'qas' field should be a list")
        if len(qas) == 0:
            raise HTTPException(status_code=400, detail="'qas' field should not be empty")
        first_qa = qas[0]

        if not isinstance(first_qa, dict):
            raise HTTPException(status_code=400, detail="Invalid 'qas' field")
        if "question" not in first_qa:
            raise HTTPException(status_code=400, detail="Missing 'question' field in the first 'qas' element")
        if not isinstance(first_qa["question"], str):
            raise HTTPException(status_code=400, detail="'question' field should be a string")
        if "answers" not in first_qa:
            raise HTTPException(status_code=400, detail="Missing 'answers' field in the first 'qas' element")
        answers = first_qa["answers"]

        if not isinstance(answers, list):
            raise HTTPException(status_code=400, detail="'answers' field should be a list")
        if len(answers) == 0:
            raise HTTPException(status_code=400, detail="'answers' field should not be empty")
        if not all(isinstance(answer, str) for answer in answers):
            raise HTTPException(status_code=400, detail="All 'answers' elements should be strings")
        
    except json.JSONDecodeError as e:
        raise HTTPException(status_code=400, detail="Invalid JSON format")