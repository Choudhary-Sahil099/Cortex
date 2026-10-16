
#include "retrival/query_transformer.hpp"

#include <cctype>
#include <string>

namespace cortex::retrieval
{

    std::string QueryTransformer::transform(const std::string &query)
    {
        std::string result;
        result.reserve(query.size());

        bool pending_space = false;

        for (unsigned char ch : query)
        {
            if (std::isspace(ch))
            {
                if (!result.empty())
                {
                    pending_space = true;
                }
                continue;
            }

            if (pending_space)
            {
                result.push_back(' ');
                pending_space = false;
            }

            if (ch >= 'A' && ch <= 'Z')
            {
                ch = static_cast<unsigned char>(ch - 'A' + 'a');
            }

            result.push_back(static_cast<char>(ch));
        }

        if (result.find("technical interview") != std::string::npos)
        {
            result += " schedule date time";
        }
        else if (
            result.find("interview") != std::string::npos &&
            result.find("attend") != std::string::npos)
        {
            result += " date time schedule";
        }
        else if (
            result.find("interview") != std::string::npos &&
            result.find("scheduled") != std::string::npos)
        {
            result += " time schedule";
        }

        return result;
    }

}