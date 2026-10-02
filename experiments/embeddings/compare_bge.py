import numpy as np
import onnxruntime as ort
from sentence_transformers import SentenceTransformer


MODEL_NAME = "BAAI/bge-small-en-v1.5"
MODEL_PATH = "models/bge-small-en-v1.5/onnx/model.onnx"


def mean_pooling(token_embeddings, attention_mask):
    mask = attention_mask[..., None]

    masked_embeddings = token_embeddings * mask

    summed = masked_embeddings.sum(axis=1)

    counts = np.clip(
        mask.sum(axis=1),
        a_min=1e-9,
        a_max=None
    )

    return summed / counts


def normalize(embeddings):
    norms = np.linalg.norm(
        embeddings,
        axis=1,
        keepdims=True
    )

    return embeddings / norms


def main():

    text = "Your technical interview is scheduled for Monday."


    #sentence refrence
    model = SentenceTransformer(MODEL_NAME)

    reference = model.encode(
        text,
        normalize_embeddings=True
    )

    #onnx
    tokenizer = model.tokenizer

    encoded = tokenizer(
        text,
        return_tensors="np"
    )

    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    outputs = session.run(
        None,
        {
            "input_ids": encoded["input_ids"].astype(np.int64),
            "attention_mask": encoded["attention_mask"].astype(np.int64),
            "token_type_ids": encoded["token_type_ids"].astype(np.int64),
        }
    )

    last_hidden_state = outputs[0]

    raw_cls = last_hidden_state[0, 0, :]

    print("Python raw CLS first 10:")

    for value in raw_cls[:10]:
        print(value)

    print("Python raw CLS norm:")
    print(np.linalg.norm(raw_cls))

    normalized_cls = raw_cls / np.linalg.norm(raw_cls)

    print("Python normalized CLS first 10:")

    for value in normalized_cls[:10]:
        print(value)

    print("ONNX output shape:")
    print(last_hidden_state.shape)

    #cls pooling

    cls_embedding = last_hidden_state[:, 0, :]

    onnx_embedding = normalize(
        cls_embedding
    )[0]

    #comparing 
    difference = np.abs(
        reference - onnx_embedding
    )

    print("\nReference first 10:")
    print(reference[:10])

    print("\nONNX first 10:")
    print(onnx_embedding[:10])

    print("\nMaximum absolute difference:")
    print(difference.max())

    print("\nMean absolute difference:")
    print(difference.mean())

    print("\nAll close:")
    print(
        np.allclose(
            reference,
            onnx_embedding,
            atol=1e-5
        )
    )


if __name__ == "__main__":
    main()