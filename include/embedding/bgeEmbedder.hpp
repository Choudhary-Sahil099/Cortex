#pragma once

#include "embedding/embedder.hpp"
#include "embedding/bert_tokenizer.hpp"

#include <onnxruntime_cxx_api.h>

#include <cstddef>
#include <string>
#include <vector>
#include <memory>


namespace cortex::embedding
{
    class BGEEmbedder : public Embedder
    {
        public:
            BGEEmbedder(const std::string &model_path,const std::string &vocab_path);
            std::vector<float> embed(const std::string &text) const override;
            std::size_t dimension() const override;
        
        private:
            static std::vector<float> normalize(std::vector<float> vector);
            Ort::Env env_;
            Ort::SessionOptions session_options_;
            std::unique_ptr<Ort::Session> session_;
            BertTokenizer tokenizer_;
            std::size_t dimension_;
    };
}