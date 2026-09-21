#pragma once

#include "core/types.hpp"
#include "vector/vector.hpp"
#include "core/metadata.hpp"
namespace cortex::core {


	// struct for vector record. Will transform it later
	struct VectorRecord {
		VectorId id;
		vector::Vector vector;
		metaData metadata;
 	};
}