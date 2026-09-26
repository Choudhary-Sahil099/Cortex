#include "email/email_cleaner.hpp"

#include <sstream> 

namespace cortex::email {
    EmailDocument EmailCleaner::clean(const EmailDocument& email) const
    {
        EmailDocument cleaned = email;

        std::istringstream stream(email.body);
        std::string line;

        cleaned.body.clear();
        bool prevEmpty = false; // checks the line is empty
        while (std::getline(stream, line)) {

            // Remove trailing whitespace
            while (!line.empty() && (line.back() == ' ' || line.back() == '\t')) {
                line.pop_back();
            }
            const bool currEmpty = line.empty();
            if (currEmpty && prevEmpty) {
                continue;
            }
            cleaned.body += line;
            cleaned.body += '\n';
            prevEmpty = currEmpty;
        }

        //removing back lines(trailing)
        while (
            cleaned.body.starts_with('\n')
            ) {
            cleaned.body.erase(0, 1);
        }
        while (
            cleaned.body.ends_with("\n\n")
            ) {
            cleaned.body.erase(
                cleaned.body.size() - 1
            );
        }
        return cleaned;
    }
}

