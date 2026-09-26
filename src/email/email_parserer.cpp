#include "email/email_parserer.hpp" 

#include <sstream>
#include <stdexcept>

namespace cortex::email {
	namespace {
		// hepler trim function to overcome the problewm of the white spaces
		std::string trim(const std::string& value)
		{
			const auto first = value.find_first_not_of(" \t");

			if (first == std::string::npos) {
				return "";
			}

			const auto last = value.find_last_not_of(" \t");

			return value.substr(first,last - first + 1);
		}
		std::vector<std::string> parseRecipients(
			const std::string& value
		)
		{
			std::vector<std::string> recipients;

			std::stringstream stream(value);
			std::string recipient;

			while (std::getline(stream, recipient, ',')) {

				recipient = trim(recipient);

				if (!recipient.empty()) {
					recipients.push_back(recipient);
				}
			}

			return recipients;
		}
	}
	EmailDocument EmailParser::parse(const std::string& raw_email) const {

		EmailDocument document;

		std::istringstream stream(raw_email);
		std::string line;

		bool in_body = false;
		std::string current_key; // for the folder struct header

		while (std::getline(stream, line)) {

			if (!line.empty() && line.back() == '\r') { // to solve the problem of the line ending with the /r
				line.pop_back();
			}
			if (!in_body) {
				if (line.empty() || line == "\r") {
					in_body = true;
					continue;
				}
				if (!line.empty() && (line.front() == ' ' || line.front() == '\t')) { // to check for the white spaces in the folder format
					if (!current_key.empty()) {
						const std::string continuation = trim(line);
						if (current_key == "Subject") {
							document.subject += " " + continuation;
						}
						else if (current_key == "Thread-ID") {
							document.thread_id += continuation;
						}
					}
					continue;
				}

				const auto separator = line.find(":"); // find the separator
				if (separator == std::string::npos) continue;
				const std::string key = line.substr(0, separator);
				current_key = key;
				std::string value = line.substr(separator + 1);

				value = trim(value); // used trim instead of the harcoded method

				if (key == "ID") document.id = value;
				else if (key == "From") document.sender = value;
				else if (key == "To") document.recipients = parseRecipients(value);
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
