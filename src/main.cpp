#include <iostream>
#include <vector>

#include "core/cpu_features.hpp"
#include "vector/backend.hpp"
#include "vector/distance.hpp"
#include "vector/dot_product.hpp"
#include "vector/similarity.hpp"
#include "vector/batch_distance.hpp"

int main()
{
    using namespace cortex::vector;

    constexpr std::size_t dimension = 1536;
    constexpr std::size_t vector_count = 1000;

    std::cout << "=================================\n";
    std::cout << "       Cortex Vector Engine\n";
    std::cout << "=================================\n\n";

    // CPU Features


    const auto features =
        cortex::core::detect_cpu_features();

    std::cout << "CPU Features\n";
    std::cout << "------------\n";
    std::cout << "AVX2 : "
        << (features.avx2 ? "supported" : "not supported")
        << '\n';

    std::cout << "FMA  : "
        << (features.fma ? "supported" : "not supported")
        << "\n\n";

    // Runtime Backend


    const auto selected_backend =
        select_distance_backend();

    std::cout << "Selected Backend\n";
    std::cout << "----------------\n";
    std::cout << backend_name(selected_backend)
        << "\n\n";

    // Create vectors
    

    Vector a(dimension);
    Vector b(dimension);

    for (std::size_t i = 0; i < dimension; ++i)
    {
        a[i] = static_cast<float>(i) * 0.001f;
        b[i] = static_cast<float>(i) * 0.002f;
    }

    std::cout << "Vector Operations\n";
    std::cout << "-----------------\n";

    std::cout << "Dimension         : "
        << dimension << '\n';

    std::cout << "L2 Distance       : "
        << l2_distance(a, b) << '\n';

    std::cout << "Dot Product       : "
        << dot_product(a, b) << '\n';

    std::cout << "Cosine Similarity : "
        << cosine_similarity(a, b) << "\n\n";
    // Batch Distance
    

    std::vector<Vector> vectors;
    vectors.reserve(vector_count);

    for (std::size_t i = 0; i < vector_count; ++i)
    {
        Vector v(dimension);

        for (std::size_t j = 0; j < dimension; ++j)
        {
            v[j] =
                static_cast<float>((i + j) % 100) / 100.0f;
        }

        vectors.push_back(std::move(v));
    }

    std::vector<float> distances;

    batch_l2_distance(
        a,
        vectors,
        distances
    );

    std::cout << "Batch Processing\n";
    std::cout << "----------------\n";

    std::cout << "Vectors           : "
        << vector_count << '\n';

    std::cout << "Dimensions/vector  : "
        << dimension << '\n';

    std::cout << "Distances computed: "
        << distances.size() << '\n';

    if (!distances.empty())
    {
        std::cout << "First distance    : "
            << distances.front() << '\n';
    }

    std::cout << "\n=================================\n";

    return 0;
}