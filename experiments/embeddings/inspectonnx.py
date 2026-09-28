import onnxruntime as ort

MODEL_PATH = "models/bge-small-en-v1.5/onnx/model.onnx"


def main():
    session = ort.InferenceSession(
        MODEL_PATH,
        providers=["CPUExecutionProvider"]
    )

    print("=== Inputs ===")

    for input_tensor in session.get_inputs():
        print("Name:", input_tensor.name)
        print("Shape:", input_tensor.shape)
        print("Type:", input_tensor.type)
        print()

    print("=== Outputs ===")

    for output_tensor in session.get_outputs():
        print("Name:", output_tensor.name)
        print("Shape:", output_tensor.shape)
        print("Type:", output_tensor.type)
        print()


if __name__ == "__main__":
    main()