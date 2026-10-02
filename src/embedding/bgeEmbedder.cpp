#include "embedding/bgeEmbedder.hpp"

#include <stdexcept>
#include <memory>
#include <array>
#include <iostream>

namespace cortex::embedding
{

    BGEEmbedder::BGEEmbedder(const std::string &model_path, const std::string &vocab_path)
        : env_(ORT_LOGGING_LEVEL_WARNING,"Cortex"),
          session_options_{},
          session_(nullptr),
          tokenizer_(vocab_path),
          dimension_(384)
    {
        session_options_.SetIntraOpNumThreads(1);

        std::wstring wide_model_path(model_path.begin(),model_path.end());

        session_ = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options_);
    }

    std::size_t BGEEmbedder::dimension() const
    {
        return dimension_;
    }

    std::vector<float> BGEEmbedder::embed(const std::string &text) const
    {

        std::cout << "BGE: starting embed()\n";

        TokenizedInput tokens = tokenizer_.encode(text);

        std::cout << "BGE: tokenization complete\n";

        const std::size_t sequence_length = tokens.input_ids.size();

        std::cout << "BGE: sequence length = " << sequence_length << "\n";

        if (sequence_length == 0)
        {
            throw std::runtime_error(
                "Tokenizer produced an empty sequence");
        }

        if (tokens.attention_mask.size() != sequence_length || tokens.token_type_ids.size() != sequence_length)
        {
            throw std::runtime_error("Tokenizer produced inconsistent input sizes");
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(sequence_length)};

        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.attention_mask.data(),
            tokens.input_ids.size(),
            input_shape.data(),
            input_shape.size());

        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.attention_mask.data(),
            tokens.attention_mask.size(),
            input_shape.data(),
            input_shape.size());

        Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.attention_mask.data(),
            tokens.token_type_ids.size(),
            input_shape.data(),
            input_shape.size());

        std::cout << "BGE: tensors created\n";
        const char *input_names[] = {
            "input_ids",
            "attention_mask",
            "token_type_ids"};

        std::array<Ort::Value, 3> input_tensors = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor),
            std::move(token_type_ids_tensor)};

        const char *output_names[] = {"last_hidden_state"};

        std::cout << "BGE: calling ONNX Run()\n";

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1);

        std::cout << "BGE: ONNX Run() completed\n";
        if (outputs.size() != 1)
        {
            throw std::runtime_error(
                "Unexpected number of ONNX outputs");
        }

        auto &output = outputs[0];

        if (!output.IsTensor())
        {
            throw std::runtime_error(
                "ONNX output is not a tensor");
        }

        const auto type_info = output.GetTensorTypeAndShapeInfo();

        const auto shape = type_info.GetShape();

        std::cout << "BGE: output shape = ";
        for (const auto value : shape)
        {
            std::cout << value << " ";
        }

        std::cout << "\n";

        if (shape.size() != 3 || shape[0] != 1 || shape[1] != static_cast<int64_t>(sequence_length) || shape[2] != static_cast<int64_t>(dimension_))
        {
            throw std::runtime_error(
                "Unexpected BGE output shape");
        }

        const float *output_data = output.GetTensorData<float>();
        std::vector<float> embedding(output_data, output_data + dimension_);

        return embedding;
    }

}