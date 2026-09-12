#include <vector/vector.hpp>
namespace cortex::vector 
{
	Vector::Vector(std::size_t dimension)
		: buffer_(dimension * sizeof(float), 32),dimension_(dimension) {} // constructor implementation

	std::size_t Vector::dimension() const 
	{
		return dimension_;// this returns the number of elements in the vector
	}
	float* Vector::data() 
	{
		return static_cast<float*>(buffer_.data()); 
	}
	const float* Vector::data() const 
	{
		return static_cast<const float*>(buffer_.data());
	}
	float& Vector::operator[](std::size_t index) {
		return data()[index];
	}
	const float& Vector::operator[](std::size_t index) const {
		return data()[index];
	}
}