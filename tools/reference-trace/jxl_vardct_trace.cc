// Copyright (c) 2026
//
// Narrow JPEG XL reference tracer for the generated split VarDCT fixture.
// This intentionally lives outside the libjxl submodule: it links against the
// locally built reference internals, uses libjxl parsing code for the container
// structure and MA tree, and mirrors libjxl's histogram bit parsing only far
// enough to report stable bit ranges for the Rust decoder oracle hook.

#include <cstddef>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "lib/jxl/ac_context.h"
#include "lib/jxl/ans_common.h"
#include "lib/jxl/ans_params.h"
#include "lib/jxl/base/bits.h"
#include "lib/jxl/base/span.h"
#include "lib/jxl/chroma_from_luma.h"
#include "lib/jxl/coeff_order_fwd.h"
#include "lib/jxl/dec_ans.h"
#include "lib/jxl/dec_bit_reader.h"
#include "lib/jxl/dec_context_map.h"
#include "lib/jxl/entropy_coder.h"
#include "lib/jxl/fields.h"
#include "lib/jxl/frame_dimensions.h"
#include "lib/jxl/frame_header.h"
#include "lib/jxl/headers.h"
#include "lib/jxl/image_metadata.h"
#include "lib/jxl/memory_manager_internal.h"
#include "lib/jxl/modular/encoding/dec_ma.h"
#include "lib/jxl/quant_weights.h"
#include "lib/jxl/quantizer.h"
#include "lib/jxl/toc.h"

namespace {

struct Section {
  size_t id = 0;
  size_t offset = 0;
  size_t size = 0;
};

struct HistogramTrace {
  size_t start = 0;
  size_t end = 0;
  bool ok = false;
};

std::vector<uint8_t> ReadFile(const char* path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) return {};
  input.seekg(0, std::ios::end);
  const std::streamoff size = input.tellg();
  if (size < 0) return {};
  input.seekg(0, std::ios::beg);
  std::vector<uint8_t> bytes(static_cast<size_t>(size));
  input.read(reinterpret_cast<char*>(bytes.data()), size);
  if (!input && size != 0) return {};
  return bytes;
}

std::string JoinContextMap(const std::vector<uint8_t>& map) {
  std::ostringstream out;
  for (size_t i = 0; i < map.size(); ++i) {
    if (i != 0) out << ",";
    out << static_cast<int>(map[i]);
  }
  return out.str();
}

int DecodeVarLenUint8(jxl::BitReader* input) {
  if (input->ReadFixedBits<1>()) {
    const int nbits = static_cast<int>(input->ReadFixedBits<3>());
    if (nbits == 0) return 1;
    return static_cast<int>(input->ReadBits(nbits)) + (1 << nbits);
  }
  return 0;
}

bool ReadHistogramForTrace(int precision_bits, jxl::BitReader* input) {
  const int range = 1 << precision_bits;
  const int simple_code = input->ReadBits(1);
  if (simple_code == 1) {
    int symbols[2] = {0};
    int max_symbol = 0;
    const int num_symbols = input->ReadBits(1) + 1;
    for (int i = 0; i < num_symbols; ++i) {
      symbols[i] = DecodeVarLenUint8(input);
      if (symbols[i] > max_symbol) max_symbol = symbols[i];
    }
    std::vector<int32_t> counts(max_symbol + 1);
    if (num_symbols == 1) {
      counts[symbols[0]] = range;
    } else {
      if (symbols[0] == symbols[1]) return false;
      counts[symbols[0]] = input->ReadBits(precision_bits);
      counts[symbols[1]] = range - counts[symbols[0]];
    }
    return input->AllReadsWithinBounds();
  }

  const int is_flat = input->ReadBits(1);
  if (is_flat == 1) {
    const int alphabet_size = DecodeVarLenUint8(input) + 1;
    return alphabet_size <= range && input->AllReadsWithinBounds();
  }

  uint32_t shift = 0;
  {
    const int upper_bound_log = jxl::FloorLog2Nonzero(ANS_LOG_TAB_SIZE + 1);
    int log = 0;
    for (; log < upper_bound_log; ++log) {
      if (input->ReadFixedBits<1>() == 0) break;
    }
    shift = (input->ReadBits(log) | (1 << log)) - 1;
    if (shift > ANS_LOG_TAB_SIZE + 1) return false;
  }

  const size_t length = DecodeVarLenUint8(input) + 3;
  std::vector<int32_t> counts(length);
  int total_count = 0;

  static const uint8_t huff[128][2] = {
      {3, 10}, {7, 12}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {7, 13}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {6, 11}, {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
      {3, 10}, {5, 0},  {3, 7}, {4, 3}, {3, 6}, {3, 8}, {3, 9}, {4, 5},
      {3, 10}, {4, 4},  {3, 7}, {4, 1}, {3, 6}, {3, 8}, {3, 9}, {4, 2},
  };

  std::vector<int> logcounts(length);
  std::vector<int> same(length);
  int omit_log = -1;
  int omit_pos = -1;
  for (size_t i = 0; i < length; ++i) {
    input->Refill();
    const int idx = input->PeekFixedBits<7>();
    input->Consume(huff[idx][0]);
    logcounts[i] = static_cast<int>(huff[idx][1]) - 1;
    if (logcounts[i] == static_cast<int>(ANS_LOG_TAB_SIZE)) {
      const int rle_length = DecodeVarLenUint8(input);
      same[i] = rle_length + 5;
      i += rle_length + 3;
      continue;
    }
    if (logcounts[i] > omit_log) {
      omit_log = logcounts[i];
      omit_pos = static_cast<int>(i);
    }
  }

  if (omit_pos < 0) return false;
  if (static_cast<size_t>(omit_pos) + 1 < length &&
      logcounts[omit_pos + 1] == static_cast<int>(ANS_LOG_TAB_SIZE)) {
    return false;
  }

  int prev = 0;
  int numsame = 0;
  for (size_t i = 0; i < length; ++i) {
    if (same[i]) {
      numsame = same[i] - 1;
      prev = i > 0 ? counts[i - 1] : 0;
    }
    if (numsame > 0) {
      counts[i] = prev;
      --numsame;
    } else {
      const int code = logcounts[i];
      if (i == static_cast<size_t>(omit_pos) || code < 0) {
        continue;
      }
      if (shift == 0 || code == 0) {
        counts[i] = 1 << code;
      } else {
        const int bitcount = jxl::GetPopulationCountPrecision(code, shift);
        counts[i] = (1 << code) +
                    (input->ReadBits(bitcount) << (code - bitcount));
      }
    }
    total_count += counts[i];
  }
  counts[omit_pos] = range - total_count;
  return counts[omit_pos] > 0 && input->AllReadsWithinBounds();
}

bool TraceResidualHistograms(JxlMemoryManager* memory_manager,
                             jxl::BitReader* br, size_t num_contexts,
                             std::vector<uint8_t>* context_map,
                             std::vector<HistogramTrace>* histograms) {
  jxl::ANSCode code;
  if (!jxl::Bundle::Read(br, &code.lz77)) return false;
  if (code.lz77.enabled) return false;

  size_t num_histograms = 1;
  context_map->assign(num_contexts, 0);
  if (num_contexts > 1) {
    if (!jxl::DecodeContextMap(memory_manager, context_map, &num_histograms,
                               br)) {
      return false;
    }
  }

  code.use_prefix_code = static_cast<bool>(br->ReadFixedBits<1>());
  if (code.use_prefix_code) {
    code.log_alpha_size = PREFIX_MAX_BITS;
  } else {
    code.log_alpha_size = br->ReadFixedBits<2>() + 5;
  }
  code.uint_config.resize(num_histograms);
  if (!jxl::DecodeUintConfigs(code.log_alpha_size, &code.uint_config, br)) {
    return false;
  }

  histograms->clear();
  histograms->reserve(num_histograms);
  for (size_t i = 0; i < num_histograms; ++i) {
    HistogramTrace trace;
    trace.start = br->TotalBitsConsumed();
    trace.ok = ReadHistogramForTrace(ANS_LOG_TAB_SIZE, br);
    trace.end = br->TotalBitsConsumed();
    histograms->push_back(trace);
    if (!trace.ok) return true;
  }
  return true;
}

bool TraceFile(const char* path) {
  std::vector<uint8_t> bytes = ReadFile(path);
  if (bytes.size() < 2 || bytes[0] != 0xff ||
      bytes[1] != jxl::kCodestreamMarker) {
    std::cerr << "expected a naked JPEG XL codestream\n";
    return false;
  }

  JxlMemoryManager memory_manager;
  if (!jxl::MemoryManagerInit(&memory_manager, nullptr)) {
    std::cerr << "failed to initialize libjxl memory manager\n";
    return false;
  }

  jxl::BitReader header_reader(
      jxl::Bytes(bytes.data() + 2, bytes.size() - 2));
  jxl::CodecMetadata metadata;
  if (!jxl::ReadSizeHeader(&header_reader, &metadata.size)) {
    std::cerr << "failed to read size header\n";
    return false;
  }
  if (!jxl::ReadImageMetadata(&header_reader, &metadata.m)) {
    std::cerr << "failed to read image metadata\n";
    return false;
  }
  metadata.transform_data.nonserialized_xyb_encoded = metadata.m.xyb_encoded;
  if (!jxl::Bundle::Read(&header_reader, &metadata.transform_data)) {
    std::cerr << "failed to read transform metadata\n";
    return false;
  }
  if (metadata.m.color_encoding.WantICC()) {
    std::cerr << "ICC-encoded metadata is not supported by this narrow tracer\n";
    return false;
  }
  if (!header_reader.JumpToByteBoundary()) {
    std::cerr << "failed to align after codestream metadata\n";
    return false;
  }
  jxl::FrameHeader frame_header(&metadata);
  if (!jxl::ReadFrameHeader(&header_reader, &frame_header)) {
    std::cerr << "failed to read frame header\n";
    return false;
  }

  const jxl::FrameDimensions frame_dim = frame_header.ToFrameDimensions();
  const size_t toc_entries = jxl::NumTocEntries(
      frame_dim.num_groups, frame_dim.num_dc_groups,
      frame_header.passes.num_passes);
  std::vector<uint32_t> sizes;
  std::vector<jxl::coeff_order_t> permutation;
  if (!jxl::ReadToc(&memory_manager, toc_entries, &header_reader, &sizes,
                    &permutation)) {
    std::cerr << "failed to read frame TOC\n";
    return false;
  }
  if (header_reader.TotalBitsConsumed() % 8 != 0) {
    std::cerr << "frame header did not end on a byte boundary\n";
    return false;
  }
  const size_t group_codes_begin = 2 + header_reader.TotalBitsConsumed() / 8;

  std::vector<Section> sections(toc_entries);
  size_t section_offset = 0;
  for (size_t i = 0; i < toc_entries; ++i) {
    const size_t physical = permutation.empty() ? i : permutation[i];
    sections[physical].id = i;
    sections[physical].offset = section_offset;
    sections[physical].size = sizes[i];
    section_offset += sizes[i];
  }

  const Section* dc_global = nullptr;
  for (const Section& section : sections) {
    if (section.id == 0) {
      dc_global = &section;
      break;
    }
  }
  if (dc_global == nullptr || group_codes_begin + dc_global->offset +
                                      dc_global->size >
                                  bytes.size()) {
    std::cerr << "failed to locate DC global section\n";
    return false;
  }

  jxl::BitReader section_reader(jxl::Bytes(bytes.data() + group_codes_begin +
                                               dc_global->offset,
                                           dc_global->size));

  jxl::DequantMatrices matrices;
  jxl::Quantizer quantizer(matrices);
  jxl::BlockCtxMap block_ctx_map;
  jxl::ColorCorrelationMap cmap;
  if (!matrices.DecodeDC(&section_reader) ||
      !quantizer.Decode(&section_reader) ||
      !jxl::DecodeBlockCtxMap(&memory_manager, &section_reader,
                              &block_ctx_map) ||
      !cmap.DecodeDC(&section_reader)) {
    std::cerr << "failed to read VarDCT global preamble\n";
    return false;
  }

  const size_t global_tree_start = section_reader.TotalBitsConsumed();
  const bool has_tree = static_cast<bool>(section_reader.ReadBits(1));
  if (!has_tree) {
    std::cerr << "DC global modular tree is not serialized\n";
    return false;
  }

  jxl::Tree tree;
  const size_t tree_size_limit =
      std::min(static_cast<size_t>(1 << 22),
               1024 + frame_dim.xsize * frame_dim.ysize * 3 / 16);
  if (!jxl::DecodeTree(&memory_manager, &section_reader, &tree,
                       tree_size_limit)) {
    std::cerr << "failed to read modular MA tree\n";
    return false;
  }
  const size_t global_tree_end = section_reader.TotalBitsConsumed();

  std::vector<uint8_t> context_map;
  std::vector<HistogramTrace> histograms;
  if (!TraceResidualHistograms(&memory_manager, &section_reader,
                               (tree.size() + 1) / 2, &context_map,
                               &histograms)) {
    std::cerr << "failed to trace residual histograms\n";
    return false;
  }

  std::cout << "global_tree_bits=" << global_tree_start << ".."
            << global_tree_end << "\n";
  std::cout << "residual_contexts=" << ((tree.size() + 1) / 2) << "\n";
  std::cout << "residual_context_map=" << JoinContextMap(context_map) << "\n";
  std::cout << "residual_histograms=" << histograms.size() << "\n";
  std::cout << "residual_histogram_bits=";
  for (size_t i = 0; i < histograms.size(); ++i) {
    if (i != 0) std::cout << ",";
    std::cout << histograms[i].start << ".." << histograms[i].end;
  }
  std::cout << "\n";
  for (size_t i = 0; i < histograms.size(); ++i) {
    if (!histograms[i].ok) {
      std::cout << "residual_histogram_error=" << i << "@"
                << histograms[i].end << "\n";
      break;
    }
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] << " INPUT.jxl\n";
    return 2;
  }
  return TraceFile(argv[1]) ? 0 : 1;
}
