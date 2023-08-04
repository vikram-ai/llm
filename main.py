import uvicorn
from llm import llm_model
from fastapi import FastAPI
m_name = "stabilityai/stablelm-base-alpha-3b"
c_model = llm_model()
app = FastAPI()

@app.post("/chat/")
def chat(user_prompt:str):
    resp = ""
    try:
        resp = c_model.generate(user_prompt=user_prompt)
    except Exception as e:
        print("Unable to Predict... Check Model")
    return {"response":resp }

if __name__ == '__main__':
    uvicorn.run("api:app", port=8800, host='0.0.0.0', reload = False, workers=1)