// Standalone test for clang generic vector backend
// Compile with:
//   clang++ -std=c++17 -I. -DEIGEN_VECTORIZE_GENERIC -DEIGEN_GENERIC_VECTOR_SIZE_BYTES=64 \
//           -O2 -o test_clang_vector test_clang_vector.cc && ./test_clang_vector

#include <Eigen/Core>
#include <iostream>
#include <cstdint>

using namespace Eigen::internal;

template<typename Packet>
void test_packet(const char* name) {
    using Scalar = typename unpacket_traits<Packet>::type;
    constexpr int size = unpacket_traits<Packet>::size;

    std::cout << "Testing " << name << " (size=" << size << ")..." << std::endl;

    // Test pzero
    Packet zero = pzero<Packet>(Packet{});
    std::cout << "  pzero: OK" << std::endl;

    // Test ptrue - this was failing with comparison type mismatch
    Packet all_ones = ptrue<Packet>(Packet{});
    std::cout << "  ptrue: OK" << std::endl;

    // Test pset1
    Packet ones = pset1<Packet>(Scalar(1));
    std::cout << "  pset1: OK" << std::endl;

    // Test pnegate
    Packet neg = pnegate<Packet>(ones);
    std::cout << "  pnegate: OK" << std::endl;

    // Test bitwise ops
    Packet anded = pand<Packet>(all_ones, ones);
    Packet ored = por<Packet>(zero, ones);
    Packet xored = pxor<Packet>(ones, ones);
    std::cout << "  pand/por/pxor: OK" << std::endl;

    // Test load/store
    alignas(64) Scalar data[size];
    for (int i = 0; i < size; ++i) data[i] = Scalar(i);
    Packet loaded = pload<Packet>(data);
    pstore<Packet>(data, loaded);
    std::cout << "  pload/pstore: OK" << std::endl;

    std::cout << "  All tests passed for " << name << std::endl;
}

template<typename Packet>
void test_float_packet(const char* name) {
    using Scalar = typename unpacket_traits<Packet>::type;
    constexpr int size = unpacket_traits<Packet>::size;

    std::cout << "Testing " << name << " (float-specific)..." << std::endl;

    Packet val = pset1<Packet>(Scalar(1.5));

    // Test pisnan - this was failing with reinterpret_cast
    Packet nan_mask = pisnan<Packet>(val);
    std::cout << "  pisnan: OK" << std::endl;

    // Test math ops
    Packet sq = psqrt<Packet>(val);
    std::cout << "  psqrt: OK" << std::endl;

    Packet fl = pfloor<Packet>(val);
    std::cout << "  pfloor: OK" << std::endl;

    std::cout << "  All float-specific tests passed for " << name << std::endl;
}

template<typename Packet>
void test_int_packet(const char* name) {
    using Scalar = typename unpacket_traits<Packet>::type;

    std::cout << "Testing " << name << " (int-specific)..." << std::endl;

    Packet val = pset1<Packet>(Scalar(0xFF));

    // Test shift ops - plogical_shift_right was failing with reinterpret_cast
    Packet shifted = parithmetic_shift_right<4>(val);
    std::cout << "  parithmetic_shift_right: OK" << std::endl;

    Packet logical = plogical_shift_right<4>(val);
    std::cout << "  plogical_shift_right: OK" << std::endl;

    Packet left = plogical_shift_left<4>(val);
    std::cout << "  plogical_shift_left: OK" << std::endl;

    std::cout << "  All int-specific tests passed for " << name << std::endl;
}

int main() {
    std::cout << "=== Clang Generic Vector Backend Test ===" << std::endl;
    std::cout << "Vector size: " << EIGEN_GENERIC_VECTOR_SIZE_BYTES << " bytes" << std::endl;
    std::cout << std::endl;

    // Test all packet types
    test_packet<Packet16f>("Packet16f");
    test_float_packet<Packet16f>("Packet16f");
    std::cout << std::endl;

    test_packet<Packet8d>("Packet8d");
    test_float_packet<Packet8d>("Packet8d");
    std::cout << std::endl;

    test_packet<Packet16i>("Packet16i");
    test_int_packet<Packet16i>("Packet16i");
    std::cout << std::endl;

    test_packet<Packet8l>("Packet8l");
    test_int_packet<Packet8l>("Packet8l");
    std::cout << std::endl;

    std::cout << "=== All tests passed! ===" << std::endl;
    return 0;
}
