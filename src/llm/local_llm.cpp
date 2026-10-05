#include "llm/local_llm.hpp"
#include <curl/curl.h>
#include <nlohmann/json.hpp>

#include <stdexcept>
#include <string>
#include <utility>

namespace cortex::llm {

namespace {

using json = nlohmann::json;

size_t writeCallback(
    char* data,
    size_t size,
    size_t count,
    void* userdata
)
{
    const size_t total_size = size * count;

    auto* response = static_cast<std::string*>(userdata);

    response->append(data, total_size);

    return total_size;
}

}

LocalLLM::LocalLLM(
    std::string server_url,
    GenerationConfig config
)
    : server_url_(std::move(server_url)),
      config_(std::move(config))
{
}

std::string LocalLLM::generate(
    const std::string& prompt
) const
{
    if (prompt.empty()) {
        throw std::invalid_argument(
            "LLM prompt cannot be empty"
        );
    }

    CURL* curl = curl_easy_init();

    if (!curl) {
        throw std::runtime_error(
            "Failed to initialize libcurl"
        );
    }

    std::string response;

    const json request = {
        {"messages", {
            {
                {"role", "user"},
                {"content", prompt}
            }
        }},
        {"max_tokens", config_.max_tokens},
        {"temperature", config_.temperature},
        {"reasoning_effort", config_.reasoning_effort}
    };

    const std::string request_body = request.dump();

    const std::string url =
        server_url_ + "/v1/chat/completions";

    struct curl_slist* headers = nullptr;

    headers = curl_slist_append(
        headers,
        "Content-Type: application/json"
    );

    curl_easy_setopt(
        curl,
        CURLOPT_URL,
        url.c_str()
    );

    curl_easy_setopt(
        curl,
        CURLOPT_POST,
        1L
    );

    curl_easy_setopt(
        curl,
        CURLOPT_HTTPHEADER,
        headers
    );

    curl_easy_setopt(
        curl,
        CURLOPT_POSTFIELDS,
        request_body.c_str()
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEFUNCTION,
        writeCallback
    );

    curl_easy_setopt(
        curl,
        CURLOPT_WRITEDATA,
        &response
    );

    const CURLcode result =
        curl_easy_perform(curl);

    if (result != CURLE_OK) {
        const std::string error =
            curl_easy_strerror(result);

        curl_slist_free_all(headers);
        curl_easy_cleanup(curl);

        throw std::runtime_error(
            "LLM request failed: " + error
        );
    }

    long http_status = 0;

    curl_easy_getinfo(
        curl,
        CURLINFO_RESPONSE_CODE,
        &http_status
    );

    curl_slist_free_all(headers);
    curl_easy_cleanup(curl);

    if (http_status < 200 || http_status >= 300) {
        throw std::runtime_error(
            "LLM server returned HTTP status " +
            std::to_string(http_status) +
            ": " +
            response
        );
    }

    try {
        const json result_json =
            json::parse(response);

        return result_json
            .at("choices")
            .at(0)
            .at("message")
            .at("content")
            .get<std::string>();
    }
    catch (const json::exception& error) {
        throw std::runtime_error(
            "Invalid LLM response: " +
            std::string(error.what())
        );
    }
}

}