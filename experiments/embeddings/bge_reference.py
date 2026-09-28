from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def main():
    model = SentenceTransformer(MODEL_NAME)

    print("Model:")
    print(model)

    print("\nModules:")
    for name, module in model.named_children():
        print(f"{name}: {module}")

    text = "Your technical interview is scheduled for Monday."

    tokenizer = model.tokenizer

    encoded = tokenizer(
        text,
        return_tensors="np"
    )

    print("\nTokenizer output:")

    for key, value in encoded.items():
        print(f"{key}:")
        print(value)
        print("shape:", value.shape)


if __name__ == "__main__":
    main()