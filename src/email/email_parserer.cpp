#include "email/email_parserer.hpp" 

#include <sstream>
#include <stdexcept>

namespace cortex::email {
	EmailDocument EmailParser::parse(const std::string& raw_email) const {

		EmailDocument document;

		std::istringstream stream(raw_email);
		std::string line;

		bool in_body = false;

		while (std::getline(stream, line)) {
			if (!in_body) {
				if (line.empty() || line == "\r") {
					in_body = true;
					continue;
				}
				const auto separator = line.find(":"); // find the separator

				if (separator == std::string::npos) continue;
				const std::string key = line.substr(0, separator);
				std::string value = line.substr(separator + 1);

				if (!value.empty() && value.front() == ' ') { 
					value.erase(0, 1);
				}

				if (key == "ID") document.id = value;
				else if (key == "From") document.sender = value;
				else if (key == "To") document.recipients.push_back(value);
				else if (key == "Subject") document.subject = value;
				else if (key == "Date") document.timestamp = value;
				else if (key == "Thread-ID") document.thread_id = value;
			}
			else {
				document.body += line;
				document.body += "\n"; // change to new line
			}
		}

		if (!document.valid()) {
			throw std::runtime_error("invalid document");
		}
		return document;
	}
	
}
