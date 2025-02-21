import os
import pickle
import uuid

import lancedb
import openai
from lancedb.embeddings import get_registry
from lancedb.pydantic import LanceModel, Vector
from lancedb.table import Table
from unstructured.chunking.basic import chunk_elements
from unstructured.partition.pdf import partition_pdf

db = lancedb.connect("db")
func = get_registry().get("openai").create(name="text-embedding-3-small")


class Embs(LanceModel):
    text: str = func.SourceField()
    vector: Vector(func.ndims()) = func.VectorField()  # type: ignore
    pg_numb: int = 0


# tbl = db.create_table("embs_v1", schema=Embs, mode="overwrite")
# tbl = db.open_table("words")
tbl = db.open_table("embs_v1")


def get_table() -> Table:
    return tbl


def extract_text_from_pdf(pdf_path) -> list[dict]:
    elements = partition_pdf(filename=pdf_path, chunking_strategy="by_title")
    elements[0].metadata.page_number
    return [
        {"text": element.text, "pg_numb": element.metadata.page_number}
        for element in elements
        if element.text.strip()
    ]


# extracted = extract_text_from_pdf("Selling-Guide_02-05-25_highlighted.pdf")


# with open("extracted_text.pkl", "wb") as f:
#     pickle.dump(extracted, f)

# with open("extracted_text.pkl", "rb") as f:
#     extracted = pickle.load(f)

# print("finished extracting !!!!!!!!")
# print(extracted[0:3])
# print(extracted[300])
# print(len(extracted))

# # Process in chunks of 100
# chunk_size = 100
# for i in range(0, len(extracted), chunk_size):
#     chunk = extracted[i : i + chunk_size]
#     print(f"Processing chunk {i // chunk_size + 1} with {len(chunk)} items...")
#     try:
#         tbl.add(chunk, on_bad_vectors="fill")  # Assuming tbl.add can handle chunks
#     except:
#         print(f"chunks @ {i} failed")
