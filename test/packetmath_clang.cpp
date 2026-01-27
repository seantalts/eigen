// This file is part of Eigen, a lightweight C++ template library
// for linear algebra.
//
// Copyright (C) 2025 Sean Talts
//
// This Source Code Form is subject to the terms of the Mozilla
// Public License v. 2.0. If a copy of the MPL was not distributed
// with this file, You can obtain one at http://mozilla.org/MPL/2.0/.

// Test for clang generic vector backend.
// This test verifies that the __builtin_bit_cast fixes for type compatibility
// issues with clang's ext_vector_type work correctly.

#include "main.h"

#ifdef EIGEN_VECTORIZE_GENERIC

#include <typeinfo>

using namespace Eigen;
using namespace Eigen::internal;

template <typename Packet>
void test_clang_packet_basics() {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int PacketSize = unpacket_traits<Packet>::size;

  EIGEN_ALIGN_MAX Scalar data1[PacketSize];
  EIGEN_ALIGN_MAX Scalar data2[PacketSize];

  // Test pzero
  Packet zero = pzero<Packet>(Packet{});
  pstore<Scalar, Packet>(data1, zero);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(0));
  }

  // Test ptrue - this was failing with comparison type mismatch
  // (long vs int64_t on some platforms)
  Packet all_ones = ptrue<Packet>(Packet{});
  pstore<Scalar, Packet>(data1, all_ones);
  // For integer types, all bits should be 1 (i.e., -1 for signed)
  // For float types, all bits 1 is NaN, so just verify it's not zero
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY(data1[i] != Scalar(0) && "ptrue should produce non-zero values");
  }

  // Test pset1
  Packet ones = pset1<Packet>(Scalar(1));
  pstore<Scalar, Packet>(data1, ones);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(1));
  }

  // Test pnegate
  Packet neg = pnegate<Packet>(ones);
  pstore<Scalar, Packet>(data1, neg);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(-1));
  }

  // Test pand
  Packet anded = pand<Packet>(all_ones, ones);
  pstore<Scalar, Packet>(data1, anded);
  pstore<Scalar, Packet>(data2, ones);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], data2[i]);
  }

  // Test por
  Packet ored = por<Packet>(zero, ones);
  pstore<Scalar, Packet>(data1, ored);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(1));
  }

  // Test pxor (ones ^ ones = zero)
  Packet xored = pxor<Packet>(ones, ones);
  pstore<Scalar, Packet>(data1, xored);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(0));
  }

  // Test load/store round-trip
  for (int i = 0; i < PacketSize; ++i) {
    data1[i] = Scalar(i);
  }
  Packet loaded = pload<Packet>(data1);
  pstore<Scalar, Packet>(data2, loaded);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], data2[i]);
  }
}

template <typename Packet>
void test_clang_packet_float() {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int PacketSize = unpacket_traits<Packet>::size;

  EIGEN_ALIGN_MAX Scalar data1[PacketSize];
  EIGEN_ALIGN_MAX Scalar data2[PacketSize];

  // Test pisnan - this was failing with reinterpret_cast
  Packet val = pset1<Packet>(Scalar(1.5));
  Packet nan_mask = pisnan<Packet>(val);
  pstore<Scalar, Packet>(data1, nan_mask);
  // 1.5 is not NaN, so mask should be all zeros
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(0));
  }

  // Test pisnan with actual NaN
  Packet nan_val = pset1<Packet>(NumTraits<Scalar>::quiet_NaN());
  nan_mask = pisnan<Packet>(nan_val);
  pstore<Scalar, Packet>(data1, nan_mask);
  // NaN should produce non-zero mask (all 1s)
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY(data1[i] != Scalar(0) && "pisnan should detect NaN");
  }

  // Test psqrt
  Packet four = pset1<Packet>(Scalar(4));
  Packet sq = psqrt<Packet>(four);
  pstore<Scalar, Packet>(data1, sq);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_APPROX(data1[i], Scalar(2));
  }

  // Test pfloor
  Packet half = pset1<Packet>(Scalar(1.5));
  Packet fl = pfloor<Packet>(half);
  pstore<Scalar, Packet>(data1, fl);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(1));
  }
}

template <typename Packet>
void test_clang_packet_int() {
  using Scalar = typename unpacket_traits<Packet>::type;
  constexpr int PacketSize = unpacket_traits<Packet>::size;

  EIGEN_ALIGN_MAX Scalar data1[PacketSize];

  Packet val = pset1<Packet>(Scalar(0xFF));

  // Test parithmetic_shift_right
  Packet shifted = parithmetic_shift_right<4>(val);
  pstore<Scalar, Packet>(data1, shifted);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(0xFF >> 4));
  }

  // Test plogical_shift_right - this was failing with reinterpret_cast
  Packet logical = plogical_shift_right<4>(val);
  pstore<Scalar, Packet>(data1, logical);
  for (int i = 0; i < PacketSize; ++i) {
    // For positive values, logical and arithmetic shift should be the same
    VERIFY_IS_EQUAL(data1[i], Scalar(0xFF >> 4));
  }

  // Test plogical_shift_left
  Packet left = plogical_shift_left<4>(val);
  pstore<Scalar, Packet>(data1, left);
  for (int i = 0; i < PacketSize; ++i) {
    VERIFY_IS_EQUAL(data1[i], Scalar(0xFF << 4));
  }

  // Test logical vs arithmetic shift with negative value
  Packet neg_val = pset1<Packet>(Scalar(-16));
  Packet arith_shift = parithmetic_shift_right<2>(neg_val);
  Packet logic_shift = plogical_shift_right<2>(neg_val);
  pstore<Scalar, Packet>(data1, arith_shift);
  VERIFY_IS_EQUAL(data1[0], Scalar(-4));  // Arithmetic shift preserves sign
}

EIGEN_DECLARE_TEST(packetmath_clang) {
  // Test float packets
  CALL_SUBTEST_1((test_clang_packet_basics<Packet16f>()));
  CALL_SUBTEST_1((test_clang_packet_float<Packet16f>()));

  // Test double packets
  CALL_SUBTEST_2((test_clang_packet_basics<Packet8d>()));
  CALL_SUBTEST_2((test_clang_packet_float<Packet8d>()));

  // Test int32 packets
  CALL_SUBTEST_3((test_clang_packet_basics<Packet16i>()));
  CALL_SUBTEST_3((test_clang_packet_int<Packet16i>()));

  // Test int64 packets
  CALL_SUBTEST_4((test_clang_packet_basics<Packet8l>()));
  CALL_SUBTEST_4((test_clang_packet_int<Packet8l>()));
}

#else  // !EIGEN_VECTORIZE_GENERIC

EIGEN_DECLARE_TEST(packetmath_clang) {
  std::cerr << "Skipping packetmath_clang test: EIGEN_VECTORIZE_GENERIC not defined" << std::endl;
}

#endif  // EIGEN_VECTORIZE_GENERIC
