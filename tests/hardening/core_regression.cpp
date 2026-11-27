#include "index/vector_index.hpp"
#include "persistence/serializer.hpp"
#include "vector/distance.hpp"
#include "vector/dot_product.hpp"
#include "vector/batch_distance_simd.hpp"
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <stdexcept>
#include <limits>
namespace fs = std::filesystem;
using cortex::index::VectorIndex;
void check(bool condition, const char* reason) { if(!condition) throw std::runtime_error(reason); }
std::size_t add(VectorIndex& index, const std::string& text, float value) {
    cortex::embedding::EmbeddedChunk chunk;
    chunk.id = std::to_string(value); chunk.text = text; chunk.index = 0;
    chunk.embedding = {value, 1.0f, 0.0f};
    return index.add(chunk);
}
struct Temp {
    fs::path path;
    Temp() { std::random_device r; path = fs::temp_directory_path()/ ("cortex-test-"+std::to_string(r())+"-"+std::to_string(r())); if(!fs::create_directory(path)) throw std::runtime_error("Test dir exists"); }
    ~Temp() { std::error_code ec; fs::remove_all(path, ec); }
};
int main() {
    try {
        VectorIndex index(3);
        auto first = add(index,"interview interview interview",1);
        auto second = add(index,"interview agenda calendar logistics notes discussion meeting",2);
        check(index.lexicalSearch("INTERVIEW!",5).front().id == first,"BM25 term frequency and length");
        check(index.lexicalSearch("no-match",3).empty(),"no match");
        auto third = add(index,"invoice payment",3);
        check(index.lexicalSearch("invoice/payment",1).front().id == third,"index invalidation on add and punctuation splitting");
        check(index.remove(first),"remove");
        check(index.lexicalSearch("interview",5).front().id == second,"index invalidation on remove");
        Temp temp; auto path = temp.path / "index.cortex";
        using cortex::persistence::Serializer;
        Serializer::save(index.hnsw(),path.string());
        VectorIndex loaded(Serializer::load(path.string()));
        check(loaded.lexicalSearch("invoice",1).front().id == third,"lexical rebuild after reload");
        add(index,"replacement document",4);
        Serializer::save(index.hnsw(),path.string());
        check(Serializer::load(path.string()).size()==3,"atomic replacement round trip");
        auto destination = temp.path / "directory"; fs::create_directory(destination);
        bool rejected=false; try { Serializer::save(index.hnsw(),destination.string()); } catch(const std::exception&) { rejected=true; }
        check(rejected && fs::is_directory(destination),"failed replace preserves destination");
        for(auto& entry:fs::directory_iterator(temp.path)) check(entry.path().filename().string().find(".cortex-save-")!=0,"staging cleanup");
        // Record count lives at byte 68 in the v2 format (magic/version + seven uint64 fields).
        { std::fstream f(path,std::ios::binary|std::ios::in|std::ios::out); f.seekp(68); auto bad=std::numeric_limits<std::uint64_t>::max(); f.write(reinterpret_cast<const char*>(&bad),sizeof bad); }
        rejected=false; try { Serializer::load(path.string()); } catch(const std::exception&) { rejected=true; }
        check(rejected,"corrupt count rejected");
        using namespace cortex::vector;
        Vector a(37),b(37); for(std::size_t i=0;i<37;++i){a[i]=float(i)*0.25f;b[i]=float(i)*0.1f;}
        auto scalar=l2_distance_scalar(a,b);
        check(std::abs(l2_distance(a,b)-scalar)<1e-4f,"runtime dispatch parity");
        check(std::abs(l2_distance_avx2(a,b)-scalar)<1e-4f,"AVX2 parity");
        check(std::abs(l2_distance_avx2_fma(a,b)-scalar)<1e-4f,"FMA parity");
        check(std::abs(dot_product(a,b)-dot_product_scalar(a,b))<1e-3f,"dot product parity");
        std::cout << "PASS: BM25, mutation/reload, atomic replacement, failure cleanup, corrupt count, SIMD parity\n";
        return 0;
    } catch(const std::exception& e) { std::cerr << "FAIL: " << e.what() << '\n'; return 1; }
}
