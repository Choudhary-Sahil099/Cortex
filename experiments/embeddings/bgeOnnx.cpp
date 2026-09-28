#include <onnxruntime_cxx_api.h>

#include <cmath>
#include <iostream>
#include <vector>
#include <array>

int main()
{
    try
    {

        // onnx runtine env
        Ort::Env env(
            ORT_LOGGING_LEVEL_WARNING,
            "Cortex-BGE");

        // create a sessiom
        Ort::SessionOptions session_options;

        session_options.SetIntraOpNumThreads(1);
        session_options.SetInterOpNumThreads(1);

        // model loading
        const wchar_t *model_path =
            L"models/bge-small-en-v1.5/onnx/model.onnx";

        Ort::Session session(
            env,
            model_path,
            session_options);

        // tokenized input

        std::vector<int64_t> input_ids = {
            101,
            2115,
            4087,
            4357,
            2003,
            5115,
            2005,
            6928,
            1012,
            102};

        std::vector<int64_t> attention_mask = {
            1, 1, 1, 1, 1,
            1, 1, 1, 1, 1};

        std::vector<int64_t> token_type_ids = {
            0, 0, 0, 0, 0,
            0, 0, 0, 0, 0};
        // shape
        std::vector<int64_t> input_shape = {
            1,
            10};

        // cpu memory info
        Ort::MemoryInfo memory_info =
            Ort::MemoryInfo::CreateCpu(
                OrtArenaAllocator,
                OrtMemTypeDefault);

        // onnx tensors
        Ort::Value input_ids_tensor =
            Ort::Value::CreateTensor<int64_t>(
                memory_info,
                input_ids.data(),
                input_ids.size(),
                input_shape.data(),
                input_shape.size());

        Ort::Value attention_mask_tensor =
            Ort::Value::CreateTensor<int64_t>(
                memory_info,
                attention_mask.data(),
                attention_mask.size(),
                input_shape.data(),
                input_shape.size());

        Ort::Value token_type_ids_tensor =
            Ort::Value::CreateTensor<int64_t>(
                memory_info,
                token_type_ids.data(),
                token_type_ids.size(),
                input_shape.data(),
                input_shape.size());

        // input names
        const char *input_names[] = {
            "input_ids",
            "attention_mask",
            "token_type_ids"};
        // output names
        const char *output_names[] = {
            "last_hidden_state"};

        // inference

        std::array<Ort::Value, 3> input_tensors = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor),
            std::move(token_type_ids_tensor)};

        std::vector<Ort::Value> outputs =
            session.Run(
                Ort::RunOptions{nullptr},
                input_names,
                input_tensors.data(),
                input_tensors.size(),
                output_names,
                1);

        Ort::Value &output = outputs[0];

        auto output_info =
            output.GetTensorTypeAndShapeInfo();

        std::vector<int64_t> output_shape =
            output_info.GetShape();

        std::cout << "Output shape:\n";

        for (const auto dimension : output_shape)
        {
            std::cout << dimension << ' ';
        }

        std::cout << "\n";

        const float *output_data =
            output.GetTensorData<float>();

        if (output_shape.size() != 3 ||
            output_shape[0] != 1 ||
            output_shape[2] != 384)
        {

            std::cerr
                << "Unexpected ONNX output shape\n";

            return 1;
        }

        const std::size_t sequence_length =
            static_cast<std::size_t>(output_shape[1]);

        const std::size_t embedding_dimension =
            static_cast<std::size_t>(output_shape[2]);

        std::vector<float> embedding(
            output_data,
            output_data + embedding_dimension);

        // l2 normalization
        float squared_norm = 0.0f;

        for (const float value : embedding)
        {
            squared_norm += value * value;
        }

        const float norm =
            std::sqrt(squared_norm);

        for (float &value : embedding)
        {
            value /= norm;
        }

        // prints first 10 values

        std::cout << "\nC++ embedding first 10:\n";

        for (std::size_t i = 0; i < 10; ++i)
        {
            std::cout << embedding[i] << '\n';
        }

        // embedding dimensions
        std::cout << "\nEmbedding dimension: "
                  << embedding.size()
                  << '\n';

        // normalization
        float final_squared_norm = 0.0f;

        for (const float value : embedding)
        {
            final_squared_norm += value * value;
        }

        std::cout << "L2 norm: "
                  << std::sqrt(final_squared_norm)
                  << '\n';

        return 0;
    }
    catch (const Ort::Exception &exception)
    {

        std::cerr
            << "ONNX Runtime error: "
            << exception.what()
            << '\n';

        return 1;
    }
}