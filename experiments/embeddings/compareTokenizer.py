from transformers import AutoTokenizer


MODEL_PATH = "models/bge-small-en-v1.5"


tokenizer = AutoTokenizer.from_pretrained(
    MODEL_PATH,
    local_files_only=True
)


test_cases = [
    "Your technical interview is scheduled for Monday.",
    "Hello, WORLD!",
    "This is an email. Please review the attached_document.pdf.",
    "John's meeting is tomorrow.",
    "technical-interview",
    "Email: TEST@example.com",
    "The number is 12345.",
    "This    has     multiple spaces.",
    "Hello\nworld\nthis is a test.",
]


for index, text in enumerate(test_cases):

    result = tokenizer(
        text,
        add_special_tokens=True,
        return_attention_mask=True,
        return_token_type_ids=True,
        truncation=False,
    )

    print(f"\n=== CASE {index} ===")
    print(f"TEXT: {text!r}")

    print("TOKENS:")
    print(tokenizer.convert_ids_to_tokens(result["input_ids"]))

    print("INPUT_IDS:")
    print(result["input_ids"])

    print("ATTENTION_MASK:")
    print(result["attention_mask"])

    print("TOKEN_TYPE_IDS:")
    print(result["token_type_ids"])