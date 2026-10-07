#include "embedding/bgeEmbedder.hpp"

#include <stdexcept>
#include <memory>
#include <array>
#include <iostream>
#include <cmath>
#include <utility>

namespace cortex::embedding
{
    //load onnx model
    BGEEmbedder::BGEEmbedder(const std::string &model_path, const std::string &vocab_path)
        : env_(ORT_LOGGING_LEVEL_WARNING, "Cortex"),
          session_options_{},
          session_(nullptr),
          tokenizer_(vocab_path),
          dimension_(384)
    {
        // std::cout << "BGE model path: "<< model_path << "\n"; // mpdel path verification
        session_options_.SetIntraOpNumThreads(1);

        std::wstring wide_model_path(model_path.begin(), model_path.end());

        session_ = std::make_unique<Ort::Session>(env_, wide_model_path.c_str(), session_options_);
        
        // std::cout << "ONNX inputs: "<< session_->GetInputCount() << "\n";

        for (std::size_t i = 0; i < session_->GetInputCount(); ++i)
        {
            auto name = session_->GetInputNameAllocated(i,Ort::AllocatorWithDefaultOptions{});

            // std::cout << "Input " << i<< ": "<< name.get()<< "\n";
        }

        // std::cout << "ONNX outputs: "<< session_->GetOutputCount()<< "\n";

        for (std::size_t i = 0; i < session_->GetOutputCount(); ++i)
        {
            auto name = session_->GetOutputNameAllocated(i,Ort::AllocatorWithDefaultOptions{});

            // std::cout << "Output " << i<< ": "<<name.get()<< "\n";
        }

        auto providers = Ort::GetAvailableProviders();

        // std::cout << "Available execution providers:\n";

        // for (const auto &provider : providers)
        // {
        //     std::cout << "  " << provider << "\n";
        // }
    }

    std::size_t BGEEmbedder::dimension() const
    {
        return dimension_;
    }



    // l2 normalization --> BGE pipeline that is validated in the python performed the L2 normalization after the CLS pooling
    std::vector<float> BGEEmbedder::normalize(std::vector<float> vector)
    {
        float squared_norm = 0.0f;

        for (const float value : vector)
        {
            squared_norm += value * value;
        }

        const float norm = std::sqrt(squared_norm);

        if (norm == 0.0f)
        {
            throw std::runtime_error("Cannot normalize zero vector");
        }

        for (float &value : vector)
        {
            value /= norm;
        }

        return vector;
    }


    std::vector<float> BGEEmbedder::embed(const std::string &text) const
    {

        // std::cout << "BGE: starting embed()\n";

        TokenizedInput tokens = tokenizer_.encode(text);

        // std::cout << "BGE: tokenization complete\n";

        const std::size_t sequence_length = tokens.input_ids.size();

        // std::cout << "BGE: sequence length = " << sequence_length << "\n";

        if (sequence_length == 0)
        {
            throw std::runtime_error("Tokenizer produced an empty sequence");
        }

        if (tokens.attention_mask.size() != sequence_length || tokens.token_type_ids.size() != sequence_length)
        {
            throw std::runtime_error("Tokenizer produced inconsistent input sizes");
        }

        Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        const std::array<int64_t, 2> input_shape = {1, static_cast<int64_t>(sequence_length)};

        //input id
        Ort::Value input_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.input_ids.data(),
            tokens.input_ids.size(),
            input_shape.data(),
            input_shape.size());


        //attension mark
        Ort::Value attention_mask_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.attention_mask.data(),
            tokens.attention_mask.size(),
            input_shape.data(),
            input_shape.size());

        //token id
        Ort::Value token_type_ids_tensor = Ort::Value::CreateTensor<int64_t>(
            memory_info,
            tokens.token_type_ids.data(),
            tokens.token_type_ids.size(),
            input_shape.data(),
            input_shape.size());

        // std::cout << "input_ids: ";

        // for (const auto id : tokens.input_ids)
        // {
        //     std::cout << id << " ";
        // }

        // std::cout << "\n";

        // std::cout << "attention_mask: ";

        // for (const auto value : tokens.attention_mask)
        // {
        //     std::cout << value << " ";
        // }

        // std::cout << "\n";

        // std::cout << "token_type_ids: ";

        // for (const auto value : tokens.token_type_ids)
        // {
        //     std::cout << value << " ";
        // }

        // std::cout << "\n";
        // std::cout << "BGE: tensors created\n";

        const char *input_names[] = {
            "input_ids",
            "attention_mask",
            "token_type_ids"};

        std::array<Ort::Value, 3> input_tensors = {
            std::move(input_ids_tensor),
            std::move(attention_mask_tensor),
            std::move(token_type_ids_tensor)};

        const char *output_names[] = {"last_hidden_state"};

        // std::cout << "BGE: calling ONNX Run()\n";

        auto outputs = session_->Run(
            Ort::RunOptions{nullptr},
            input_names,
            input_tensors.data(),
            input_tensors.size(),
            output_names,
            1);

        // std::cout << "BGE: ONNX Run() completed\n";

        if (outputs.size() != 1)
        {
            throw std::runtime_error("Unexpected number of ONNX outputs");
        }

        auto &output = outputs[0];

        if (!output.IsTensor())
        {
            throw std::runtime_error("ONNX output is not a tensor");
        }

        const auto type_info = output.GetTensorTypeAndShapeInfo();

        const auto shape = type_info.GetShape();

        // std::cout << "BGE: output shape = ";
        // for (const auto value : shape)
        // {
        //     std::cout << value << " ";
        // }

        // std::cout << "\n";

        if (shape.size() != 3 || shape[0] != 1 || shape[1] != static_cast<int64_t>(sequence_length) || shape[2] != static_cast<int64_t>(dimension_))
        {
            throw std::runtime_error(
                "Unexpected BGE output shape");
        }

        const float *output_data = output.GetTensorData<float>();
        std::vector<float> embedding(output_data, output_data + dimension_);

        // debugging
        float raw_squared_norm = 0.0f;

        for (const float value : embedding)
        {
            raw_squared_norm += value * value;
        }

        // std::cout
        //     << "Raw CLS norm = "
        //     << std::sqrt(raw_squared_norm)
        //     << "\n";

        // std::cout << "Raw CLS first 10:\n";

        // for (std::size_t i = 0; i < 10; ++i)
        // {
        //     std::cout << embedding[i] << "\n";
        // }

        auto normalized = normalize(std::move(embedding));

        // std::cout << "Normalized first 10:\n";

        // for (std::size_t i = 0; i < 10; ++i)
        // {
        //     std::cout << normalized[i] << "\n";
        // }

        return normalized;
    }

}