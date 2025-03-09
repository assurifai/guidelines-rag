import openai
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from ingest import Embs, get_table

load_dotenv()

# Load the table for embedding search
tbl = get_table()

# Initialize FastAPI app
app = FastAPI(title="RAG Chatbot API")

# OpenAI client
client = openai.OpenAI()


class ChatRequest(BaseModel):
    query: str
    num_contexts: int = 3


class DetailedContext(BaseModel):
    text: str
    pg_numb: int


class ChatResponse(BaseModel):
    response: str
    sources: list[DetailedContext]


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        results: list[Embs] = (
            tbl.search(request.query).limit(request.num_contexts).to_pydantic(Embs)
        )
        retrieved_texts = [result.text for result in results]
        detailed_contexts = [
            DetailedContext(text=emb.text, pg_numb=emb.pg_numb) for emb in results
        ]
        messages = [
            {
                "role": "system",
                "content": "You are a RAG Chatbot for a lender. The relevant context is pulled from Fannie Mae guidelines, answer the questions based on Fannie Mae's guidelines.",
            },
            {"role": "user", "content": request.query},
            {
                "role": "user",
                "content": "Relevant context: " + "\n\n".join(retrieved_texts),
            },
        ]

        response = client.chat.completions.create(
            model="gpt-4o-mini", messages=messages, stream=False
        )

        response_text = response.choices[0].message.content

        return ChatResponse(response=response_text, sources=detailed_contexts)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
